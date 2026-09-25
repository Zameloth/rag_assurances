"""Runs one rung/arm of the retrieval ladder through `dataset.run_experiment()` and persists
the result — SPEC §12.5/§12.11/§12.12, ADR-0011, ADR-0012, #35.

**Paired with @Zameloth** (see the #35 issue comment and #28/#31's precedent).

**Forcing tracing on, resolved**: `dataset.run_experiment()` is a thin wrapper —
`DatasetClient.run_experiment` just calls `self._langfuse_client.run_experiment(...)`
(`langfuse/_client/datasets.py`) — so *which* `Langfuse` instance's tracing is "forced on"
is exactly the instance `get_dataset` was called on, not some other ambient client. That
means forcing `LANGFUSE_TRACING` true "whatever `rag.config.Settings.langfuse_tracing`'s dev
value is" (ADR-0012) doesn't need an env-var override or the `get_client()` singleton at
all: this module constructs its own `Langfuse(tracing_enabled=True, ...)` client and never
reads `Settings.langfuse_tracing` — the dev flag simply never enters this call's decision.
`LANGFUSE_BASE_URL` and the two keys still come from `rag.config.load_settings()`, never a
raw env var, per the other acceptance criterion.

**`langfuse_dataset_version`, resolved**: `DatasetClient.version` only holds whatever
`version=` was *passed into* `get_dataset` — `None` when, as here, the latest version is
wanted rather than a pinned historical one. `DatasetClient.updated_at` is what actually
moves on every dataset change (SPEC's "every add/update/delete/archive produces a new
version", and versions are timestamps) — see `docs/research/langfuse-rag-eval.md` §2.2 —
so that, not `.version`, is what the header records.

**Per-item scores, resolved**: `ExperimentItemResult.output` is the exact Python object the
`task` returned (in-process, not round-tripped through JSON) — SPEC §12.11's per-item rows
are recomputed straight from it after `run_experiment()` returns, rather than parsed back
out of the `Evaluation`s the evaluator emitted for the Langfuse UI. Same math, two audiences:
the evaluator's `Evaluation`s are for the traces, this module's own `score_item` call is for
the JSON.
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

from rag.config import load_settings
from rag.eval.langfuse_sync import RETRIEVAL_DATASET_NAME, reconstruct_golden_item
from rag.eval.retrieval_metrics import ItemRetrievalScore, score_item
from rag.eval.retrieval_run import RetrievalRun, RunHeader, resolve_git_sha, write_run
from rag.eval.schema import GoldenItem
from rag.ingest.upsert import EmbedFn
from rag.retrieval.pipeline import RetrievalResult, retrieve

__all__ = ["MAX_CONCURRENCY", "MIN_CONCURRENCY", "run_retrieval_ladder"]

# SPEC §12.5/ADR-0012 — "the SDK default of 50 will rate-limit OpenRouter". Named as a
# range rather than a single constant since #35's acceptance criterion itself says 5-10,
# not one fixed number.
MIN_CONCURRENCY = 5
MAX_CONCURRENCY = 10

# The fields of `ItemRetrievalScore` that are actual metrics — everything but the id and
# the short-circuit path, which are identifying/diagnostic, not a number to score on.
_SCORE_FIELDS = (
    "fiche_recall_at_4",
    "fiche_recall_at_10",
    "fiche_recall_at_candidate",
    "article_recall_at_4",
    "article_recall_at_10",
    "article_recall_at_candidate",
    "zero_articles",
    "floor_correct",
    "span_containment_at_4",
)
_BOOLEAN_SCORE_FIELDS = frozenset({"zero_articles", "floor_correct"})


def _score_evaluations(score: ItemRetrievalScore) -> list[Evaluation]:
    """One `Evaluation` per non-`None` metric in `score`, so a metric that's undefined for
    this item (SPEC §12.6: empty gold, off the `reponse_sans_article` subset, or the
    short-circuit's empty pool) is simply absent from the trace rather than showing up as a
    misleading zero."""
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


def _golden_item_from(item: DatasetItem) -> GoldenItem:
    """`score_item` needs a `GoldenItem`; a `run_experiment()` dataset item only ever
    carries what `rag.eval.langfuse_sync.retrieval_dataset_items` put there. `item.id` is
    the golden id verbatim — `dataset_item_id` is the identity function (see its own
    docstring) — so there is no need to thread it through `item.metadata` separately."""
    return reconstruct_golden_item(golden_id=item.id, question=str(item.input), expected_output=item.expected_output)


def run_retrieval_ladder(
    *,
    client: QdrantClient,
    embed: EmbedFn,
    lookup_keys: AbstractSet[str],
    arm: str,
    rung: str,
    run_id: str,
    repo_root: Path,
    golden_set_path: Path,
    runs_dir: Path,
    retrieval_config: dict[str, Any],
    dataset_name: str = RETRIEVAL_DATASET_NAME,
    pipeline_arm: str | None = None,
    max_concurrency: int = MAX_CONCURRENCY,
) -> RetrievalRun:
    """Run `arm` against the synced `dataset_name` dataset and persist the result to
    `runs_dir` (SPEC §12.11's `eval/runs/<run-id>.json`).

    `retrieval_config` is the caller's own pinned description of what this rung/arm actually
    runs (embedder id, chunker params, per-leg top-k, fusion weights, rerank on/off — SPEC
    §12.11) — pulled from `rag.retrieval.pipeline`'s named constants for whichever rung this
    is, not reconstructed here, since only the caller knows which rung it asked for.

    **`pipeline_arm` decouples "which `rag.retrieval.pipeline.RETRIEVAL_ARMS` entry runs"
    from "what `RunHeader.arm` records" (SPEC §12.8, #38).** For rungs 1-5 the two are the
    same thing and `pipeline_arm` stays `None` (falling back to `arm`) — unchanged from #35.
    The two pre-ladder A/Bs need them to differ: both `ab_article_breadcrumb`'s incumbent and
    challenger runs exercise the *same* fixed hybrid pipeline (`pipeline_arm="rung2"`, the
    earliest config that actually uses both vector kinds — SPEC §9.3), while `arm` records
    which collection served the query (`"incumbent"`/`"challenger"`, flipped in by the
    caller before each run) — a fact `RETRIEVAL_ARMS` has no entry for, since it is a
    property of which Qdrant collection the stable alias points at, not of the pipeline.

    A fresh, explicitly-constructed `Langfuse` client is used rather than the `get_client()`
    singleton — see the module docstring's "forcing tracing on" note — and it, not
    `get_client()`'s ambient instance, is what `dataset.run_experiment()` runs through.

    **`max_concurrency` defaults to `MAX_CONCURRENCY`** (#35's original range, unchanged for
    every existing caller), but is a real parameter — #41's rung 4 onward is the first arm
    whose `task` does local, CPU-bound cross-encoder inference rather than an API call:
    `MIN_CONCURRENCY`/`MAX_CONCURRENCY`'s own "the SDK default of 50 will rate-limit
    OpenRouter" reasoning is about *API* concurrency, and running several reranker forward
    passes at once competes for the same CPU/RAM a single process already needs for one
    (SPEC §14.4's own "hundreds of MB" of transient activations per rerank call, times
    however many run at once) — a caller doing that work asks for a lower number instead of
    a code change here.
    """
    settings = load_settings()
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        tracing_enabled=True,
    )
    dataset = langfuse.get_dataset(dataset_name)
    resolved_pipeline_arm = pipeline_arm if pipeline_arm is not None else arm

    def task(*, item: ExperimentItem, **kwargs: Any) -> RetrievalResult:
        # `dataset.run_experiment()` only ever hands the task a `DatasetItem` (the
        # `LocalExperimentItem` half of the union is `langfuse.run_experiment(data=[...])`'s
        # own local-data path, never this dataset-bound one) — asserted, not just cast, so a
        # future SDK change surfaces here rather than as a silent `AttributeError` downstream.
        assert isinstance(item, DatasetItem)
        return retrieve(client, embed, str(item.input), lookup_keys, arm=resolved_pipeline_arm)

    def retrieval_evaluator(
        *, input: Any, output: RetrievalResult, expected_output: Any, metadata: Any, **kwargs: Any
    ) -> list[Evaluation]:
        # `run_experiment()` only ever hands evaluators `input`/`output`/`expected_output`/
        # `metadata` (`client.py`'s own evaluator-invocation code, not the `item` the task
        # got) — so the golden id has to come from `metadata["golden_id"]`
        # (`retrieval_dataset_items` put it there), not `item.id`.
        golden_item = reconstruct_golden_item(
            golden_id=metadata["golden_id"], question=str(input), expected_output=expected_output
        )
        return _score_evaluations(score_item(golden_item, output))

    result = dataset.run_experiment(
        name=run_id,
        run_name=run_id,
        task=task,
        evaluators=[retrieval_evaluator],
        max_concurrency=max_concurrency,
    )

    items = tuple(
        score_item(_golden_item_from(item_result.item), item_result.output)
        for item_result in result.item_results
        if isinstance(item_result.item, DatasetItem)
    )
    header = RunHeader(
        run_id=run_id,
        rung=rung,
        arm=arm,
        golden_set_git_sha=resolve_git_sha(repo_root, path=golden_set_path),
        langfuse_dataset_version=dataset.updated_at.isoformat(),
        retrieval_config=retrieval_config,
        code_git_sha=resolve_git_sha(repo_root),
        timestamp=datetime.now(UTC).isoformat(),
        langfuse_run_name=run_id,
    )
    run = RetrievalRun(header=header, items=items)
    write_run(run, runs_dir)
    langfuse.flush()
    return run
