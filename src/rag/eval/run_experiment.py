"""Runs one rung/arm of the retrieval ladder through `dataset.run_experiment()` and persists
the result — SPEC §12.5/§12.11/§12.12, ADR-0011, ADR-0012, #35.

**Paired with @Zameloth** (see the #35 issue comment and #28/#31's precedent): this file is
a skeleton. Everything it hands off to — `rag.retrieval.pipeline.retrieve`,
`rag.eval.retrieval_metrics.score_item`, `rag.eval.retrieval_run.RunHeader`/`RetrievalRun`/
`write_run`/`resolve_git_sha`, `rag.eval.langfuse_sync.dataset_item_id` — already exists and
is tested. What's missing is the actual `langfuse` SDK wiring: the `task` and evaluator
functions `dataset.run_experiment()` calls, and forcing this run's tracing/concurrency
settings regardless of `rag.config`'s dev defaults. See the TODO inside
`run_retrieval_ladder` below for the seams to fill in together.

Fixed constraints (#35's own acceptance criteria — these are settled, just need wiring in):
`max_concurrency` must be in `[MIN_CONCURRENCY, MAX_CONCURRENCY]` (5-10) — the SDK default
of 50 exists to rate-limit an LLM provider, which is moot here since the task below makes no
LLM call at all (ADR-0012: "a ladder run costs no API spend by construction"), but the
acceptance criterion still asks for it set explicitly rather than left at the default.
`LANGFUSE_TRACING` must be forced `True` for the duration of this call, whatever
`rag.config.Settings.langfuse_tracing`'s dev value is (ADR-0012). Read `LANGFUSE_BASE_URL`
through `rag.config.load_settings()`, never a raw env var — `Settings.from_env` already
rejects the dead `LANGFUSE_HOST` name outright, so this file only ever sees the right one.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from rag.eval.langfuse_sync import RETRIEVAL_DATASET_NAME
from rag.eval.retrieval_run import RetrievalRun
from rag.ingest.upsert import EmbedFn

__all__ = ["MAX_CONCURRENCY", "MIN_CONCURRENCY", "run_retrieval_ladder"]

# SPEC §12.5/ADR-0012 — "the SDK default of 50 will rate-limit OpenRouter". Named as a
# range rather than a single constant since #35's acceptance criterion itself says 5-10,
# not one fixed number.
MIN_CONCURRENCY = 5
MAX_CONCURRENCY = 10


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
) -> RetrievalRun:
    """Run `arm` against the synced `dataset_name` dataset and persist the result to
    `runs_dir` (SPEC §12.11's `eval/runs/<run-id>.json`).

    `retrieval_config` is the caller's own pinned description of what this rung/arm actually
    runs (embedder id, chunker params, per-leg top-k, fusion weights, rerank on/off — SPEC
    §12.11) — pulled from `rag.retrieval.pipeline`'s named constants for whichever rung this
    is, not reconstructed here, since only the caller knows which rung it asked for.

    TODO (Langfuse-specific — pair on this):

    1. `langfuse = get_client()`; `dataset = langfuse.get_dataset(dataset_name)`. Record
       `dataset`'s version (the research doc's §2.2 has the accessor) for
       `RunHeader.langfuse_dataset_version` — the run header's join back to *which* sync of
       the golden set this run actually scored.
    2. Force tracing on for this call: `LANGFUSE_TRACING` must read `True` regardless of
       `rag.config.load_settings().langfuse_tracing`'s dev value (ADR-0012). Check the SDK
       for the right way to do a scoped override (an env var flip around the call, a
       `Langfuse(...)` constructor kwarg, or something `get_client()` itself takes) rather
       than assuming — this is exactly the kind of thing the research doc's "verify against
       source, the docs lag" warning is about.
    3. Write the `task` function: given one Langfuse dataset item, call
       `rag.retrieval.pipeline.retrieve(client, embed, item.input, lookup_keys, arm=arm)`
       and return the `RetrievalResult` (or an equivalent mapping) — evaluators only see the
       task's return value, never the trace (ADR-0012), and `score_item` needs the whole
       `RetrievalResult`, not just `contexts`.
    4. Write the evaluator(s): reconstruct the original `GoldenItem` from the dataset item
       (its `expected_output`/`metadata` already carry every field `score_item` reads —
       `rag.eval.langfuse_sync.retrieval_dataset_items` fixes that shape), call
       `score_item(golden_item, retrieval_result)`, and turn each `ItemRetrievalScore` field
       into a Langfuse `Evaluation` so the numbers also show up on the traces, not only in
       the persisted JSON.
    5. Call `dataset.run_experiment(name=..., run_name=run_id, task=..., evaluators=...,
       max_concurrency=<pick one in [MIN_CONCURRENCY, MAX_CONCURRENCY]>)`.
    6. Turn the experiment's per-item results into `tuple[ItemRetrievalScore, ...]` — either
       read them back off the `ExperimentResult`, or recompute them directly from each
       task's `RetrievalResult` (decide which is more direct once the shape of
       `ExperimentResult` is in front of us).
    7. Assemble `RunHeader(run_id=run_id, rung=rung, arm=arm,
       golden_set_git_sha=resolve_git_sha(repo_root, path=golden_set_path),
       code_git_sha=resolve_git_sha(repo_root), langfuse_dataset_version=..., timestamp=...,
       retrieval_config=retrieval_config, langfuse_run_name=run_id)` — `langfuse_run_name`
       is the same string passed as `run_name` in step 5, so the header is one click from
       the traces (SPEC §12.11).
    8. `write_run(RetrievalRun(header, items), runs_dir)` and return it.
    """
    raise NotImplementedError
