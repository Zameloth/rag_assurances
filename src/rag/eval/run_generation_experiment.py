"""Runs one arm of the generation eval through `dataset.run_experiment()` and persists the
result — SPEC §12.5/§12.9/§12.11/§12.12, ADR-0009, #46.

Mirrors `rag.eval.run_experiment`'s own shape one stage further down the pipeline: a task
closure that this time runs the **full chain** — condensation, then retrieval, then
generation — rather than retrieval alone, because the generation dataset's ten `multi_turn`
items are the only golden-set items that measure the condenser at all (SPEC §12.1), and
`generate()` never sees a condensed query itself (ADR-0008): only the query handed to
`retrieve()` is condensed, `generate()` always gets the raw current turn back.

**Forcing tracing on and the fresh `Langfuse` client, same reasoning as
`rag.eval.run_experiment`'s own module docstring** — this module constructs its own
`Langfuse(tracing_enabled=True, ...)` instance rather than reading `Settings.langfuse_tracing`
or reaching for `get_client()`'s ambient singleton, and `dataset.run_experiment()` runs
through that instance, not some other one.

**Only the three deterministic metrics are evaluators here** (SPEC §12.9): state accuracy,
citation validity, citation correctness. Point coverage and faithfulness are judged metrics
a later ticket adds as further evaluators over the same `task` output — this module's own
per-item persistence (`ItemGenerationScore`) has no fields for them yet, on purpose, so
their absence isn't threatened by a wrong guess at a judge's shape made here.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langfuse import Evaluation, Langfuse
from langfuse.api import DatasetItem
from langfuse.experiment import ExperimentItem
from qdrant_client import QdrantClient

from rag.condensation.pipeline import CondenseFn, condense
from rag.condensation.prompt import HistoryTurn as CondensationHistoryTurn
from rag.config import load_settings
from rag.eval.generation_metrics import ItemGenerationScore, score_item
from rag.eval.generation_run import GenerationRun, write_generation_run
from rag.eval.langfuse_sync import GENERATION_DATASET_NAME, reconstruct_generation_item
from rag.eval.retrieval_run import RunHeader, resolve_git_sha
from rag.eval.schema import GoldenItem
from rag.generation.pipeline import GenerateFn, GenerationResult, generate
from rag.generation.prompt import HistoryTurn as GenerationHistoryTurn
from rag.ingest.upsert import EmbedFn
from rag.retrieval.pipeline import DEFAULT_RETRIEVAL_ARM, retrieve

__all__ = ["GENERATION_RUNG", "MAX_CONCURRENCY", "MIN_CONCURRENCY", "run_generation_eval"]

# Same range and reasoning as `rag.eval.run_experiment` — every generation-eval item makes
# two paid OpenRouter calls instead of retrieval's zero (condensation, when history is
# non-empty, and generation always), so the SDK's default-50 concurrency would rate-limit
# harder here, not less.
MIN_CONCURRENCY = 5
MAX_CONCURRENCY = 10

# `RunHeader.rung` records which experiment axis is under test (SPEC §12.11) — the ladder
# spends it on rung/A-B names because six rungs and two pre-ladder A/Bs are the axis; the
# generation regime has one axis so far, the ablatable generation model itself
# (`RunHeader.arm`, `RunHeader.generation_model`/`generation_provider`), so `rung` is this
# one constant rather than a per-call parameter with nothing yet to vary it.
GENERATION_RUNG = "generation"

_SCORE_FIELDS = ("state_correct", "citation_valid", "citation_correctness")
_BOOLEAN_SCORE_FIELDS = frozenset({"state_correct", "citation_valid"})


def _score_evaluations(score: ItemGenerationScore) -> list[Evaluation]:
    """One `Evaluation` per non-`None` deterministic metric in `score` — `citation_correctness`
    is `None` on an item with no `gold_articles` (SPEC §12.9's "undefined, not zero" posture,
    same as `state_correct`/`citation_valid` never being anything but a real bool)."""
    evaluations = []
    for field_name in _SCORE_FIELDS:
        value = getattr(score, field_name)
        if value is None:
            continue
        data_type: Literal["NUMERIC", "BOOLEAN"] = (
            "BOOLEAN" if field_name in _BOOLEAN_SCORE_FIELDS else "NUMERIC"
        )
        evaluations.append(Evaluation(name=field_name, value=value, data_type=data_type))
    return evaluations


def _history_from_metadata(metadata: Any) -> tuple[CondensationHistoryTurn, ...]:
    """`generation_dataset_items` puts `history` in `metadata` as plain `{"role", "content"}`
    dicts (JSON round-trips through Langfuse as exactly that, never a richer type) — this is
    the one place that turns them back into `HistoryTurn`s, condensation's own type since
    it's the first stage of the chain to need one."""
    return tuple(CondensationHistoryTurn(role=turn["role"], content=turn["content"]) for turn in metadata.get("history", []))


def _golden_item_from(item: DatasetItem) -> GoldenItem:
    return reconstruct_generation_item(
        golden_id=item.id, question=str(item.input), expected_output=item.expected_output, metadata=item.metadata
    )


def run_generation_eval(
    *,
    client: QdrantClient,
    embed: EmbedFn,
    lookup_keys: AbstractSet[str],
    condense_fn: CondenseFn,
    generate_fn: GenerateFn,
    arm: str,
    run_id: str,
    repo_root: Path,
    golden_set_path: Path,
    runs_dir: Path,
    generation_model: str,
    generation_provider: str,
    retrieval_config: dict[str, Any],
    dataset_name: str = GENERATION_DATASET_NAME,
    retrieval_arm: str = DEFAULT_RETRIEVAL_ARM,
    max_concurrency: int = MAX_CONCURRENCY,
) -> GenerationRun:
    """Run the full chain (condense -> retrieve -> generate) against every item in the
    synced `dataset_name` dataset and persist the result to `runs_dir` (SPEC §12.11's
    `eval/runs/<run-id>.json`).

    `condense_fn`/`generate_fn` are the same injection seam `rag.eval.run_experiment` takes
    for `embed` — the real chains come from `rag.condensation.chain.make_condense_fn` /
    `rag.generation.chain.make_generate_fn`, a test injects fakes. `arm` names the
    generation-model arm for `RunHeader.arm` (SPEC §10.1: "the generation model is an
    ablatable arm"); `generation_model`/`generation_provider` are the caller's own pinned
    description of what actually ran, since only the caller (which built `generate_fn`)
    knows which `Settings` it was built from. `retrieval_arm` defaults to
    `DEFAULT_RETRIEVAL_ARM` — the ladder-winning arm, once adopted, is what a generation run
    should sit on top of, not whichever rung happened to run last.

    **`generation_provider` pins the requested provider, not a response-verified one.**
    SPEC §12.10 asks for "the resolved provider recorded in every persisted run" for the
    *judge* call, reasoning that OpenRouter's own routing is itself a variance source; the
    same would ideally hold here. But `GenerateFn` (`rag.generation.pipeline`) returns a
    bare `Envelope`, and `rag.generation.chain`'s `with_structured_output()` call (no
    `include_raw=True`) gives it nothing else to return: checked against the installed
    `langchain_openai`, `ChatOpenAI._create_chat_result` builds `response_metadata` from a
    fixed key allowlist (`token_usage`, a hardcoded `model_provider: "openai"`, `model_name`,
    `system_fingerprint`, `id`, `service_tier`) and never forwards OpenRouter's own `provider`
    response field. Reading that back for real needs `include_raw=True` plus a `GenerateFn`
    signature change threading response metadata through — out of scope here, and exactly
    the "confirm against the actual HTTP body" gap `rag.generation.chain`'s own module
    docstring already leaves open. What *is* pinned is trustworthy for a successful call:
    `allow_fallbacks: false` (SPEC §10.1) means OpenRouter had nowhere else to route it —
    the requested provider is the only one that could have served a response that came back
    at all.
    """
    settings = load_settings()
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        tracing_enabled=True,
    )
    dataset = langfuse.get_dataset(dataset_name)

    def task(*, item: ExperimentItem, **kwargs: Any) -> GenerationResult:
        # Same assertion `rag.eval.run_experiment.task` makes, same reason: a dataset-bound
        # `run_experiment()` call only ever hands its task a `DatasetItem`.
        assert isinstance(item, DatasetItem)
        raw_turn = str(item.input)
        condensation_history = _history_from_metadata(item.metadata)

        condensation = condense(raw_turn, condensation_history, lookup_keys, condense_fn)
        retrieval = retrieve(client, embed, condensation.query, lookup_keys, arm=retrieval_arm)

        generation_history = tuple(
            GenerationHistoryTurn(role=turn.role, content=turn.content) for turn in condensation_history
        )
        return generate(raw_turn, retrieval, generate_fn, history=generation_history)

    def generation_evaluator(
        *, input: Any, output: GenerationResult, expected_output: Any, metadata: Any, **kwargs: Any
    ) -> list[Evaluation]:
        golden_item = reconstruct_generation_item(
            golden_id=metadata["golden_id"], question=str(input), expected_output=expected_output, metadata=metadata
        )
        return _score_evaluations(score_item(golden_item, output))

    result = dataset.run_experiment(
        name=run_id,
        run_name=run_id,
        task=task,
        evaluators=[generation_evaluator],
        max_concurrency=max_concurrency,
    )

    items = tuple(
        score_item(_golden_item_from(item_result.item), item_result.output)
        for item_result in result.item_results
        if isinstance(item_result.item, DatasetItem)
    )
    header = RunHeader(
        run_id=run_id,
        rung=GENERATION_RUNG,
        arm=arm,
        golden_set_git_sha=resolve_git_sha(repo_root, path=golden_set_path),
        langfuse_dataset_version=dataset.updated_at.isoformat(),
        retrieval_config=retrieval_config,
        code_git_sha=resolve_git_sha(repo_root),
        timestamp=datetime.now(UTC).isoformat(),
        langfuse_run_name=run_id,
        generation_model=generation_model,
        generation_provider=generation_provider,
    )
    run = GenerationRun(header=header, items=items)
    write_generation_run(run, runs_dir)
    langfuse.flush()
    return run
