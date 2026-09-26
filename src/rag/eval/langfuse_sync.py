"""One-directional sync: `eval/golden/golden-set.yaml` -> two Langfuse datasets,
`rag-assurances-retrieval` (SPEC §12.5, #35) and `rag-assurances-generation` (SPEC §12.5,
§12.9, #46), by the same id-keyed script.

**Everything in this module is plain data-shaping** — no Langfuse import, no network call —
so it is implemented and tested here like the rest of the harness. `sync_retrieval_dataset`
was originally paired with @Zameloth (#35 issue comment, the precedent #28/#31 already set for
`rag.retrieval.langchain_retriever`'s `_traced`); `sync_generation_dataset` below reuses the
same two SDK findings that pairing settled (next paragraph) rather than re-deriving them, per
#46's own issue comment choosing agent-authored over paired for this ticket.

Two calls into the SDK source (langfuse==4.15.1) settled what the docs don't say: `get_dataset`
re-raises `langfuse.api.NotFoundError` on an unknown name (so get-or-create is a plain
try/except, no need to risk `create_dataset`'s own idempotency on re-sync), and
`create_dataset_item(id=...)` "upserts if an item with id already exists" per its own
docstring — no fetch-then-update path needed. A removed golden-set item is left as a
deliberate orphan (`rag.eval.ids`'s ids are never reused, so it can never collide with a
future item) rather than archived — the simpler of the two options and a deliberate call,
not a placeholder.

**A third finding, from an actual live sync (#46): item ids are unique per Langfuse project,
not per dataset.** Pushing the generation dataset under bare golden ids (`dataset_item_id`,
already live in `rag-assurances-retrieval`) failed with `LangfuseConflictError` — "item ids
are unique per project across datasets". `generation_dataset_item_id` exists because of
exactly this: a second, disjoint id namespace for the generation dataset, even though the two
datasets share every golden id as ground truth.

**The generation dataset carries the full 60 items, not the working set** — SPEC §12.5's
"two datasets, two regimes": the ladder runs the `history == []` retrieval-bearing subset
only, but generation eval scores every terminal state (refusals and `hors_corpus` included)
and both single- and multi-turn items, since the ten `multi_turn` items are the only ones
that measure the condenser at all (SPEC §12.1). `history` travels in `metadata` rather than
`input` — `rag.eval.run_generation_experiment`'s task needs it to run condensation before
retrieval, and `dataset.run_experiment()` only ever hands a task/evaluator the four fields
`retrieval_dataset_items` already established the shape for (`input`, `output`,
`expected_output`, `metadata`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langfuse import get_client
from langfuse.api import NotFoundError

from rag.eval.retrieval_metrics import working_set
from rag.eval.schema import GoldenItem, load_golden_set

__all__ = [
    "GENERATION_DATASET_NAME",
    "RETRIEVAL_DATASET_NAME",
    "GenerationDatasetItem",
    "RetrievalDatasetItem",
    "dataset_item_id",
    "generation_dataset_item_id",
    "generation_dataset_items",
    "reconstruct_generation_item",
    "reconstruct_golden_item",
    "retrieval_dataset_items",
    "sync_generation_dataset",
    "sync_retrieval_dataset",
]

RETRIEVAL_DATASET_NAME = "rag-assurances-retrieval"
GENERATION_DATASET_NAME = "rag-assurances-generation"


def dataset_item_id(golden_id: str) -> str:
    """The Langfuse dataset item id for a golden-set item in the **retrieval** dataset —
    golden ids are already SPEC §12.1's stable, hand-assigned, never-renumbered natural key,
    so reusing one verbatim is the simplest deterministic derivation and is what makes this
    sync idempotent (#35: "an id-keyed sync script"): re-running it against an unchanged
    item updates the same dataset item instead of minting a duplicate, and every arm's
    `run_experiment` run joins its `DatasetRunItem` back to the same item regardless of
    which rung produced it.

    **Confirmed live, #46**: dataset item ids are unique *per Langfuse project*, not per
    dataset — pushing a bare golden id already used here into `rag-assurances-generation`
    fails with `LangfuseConflictError` ("item ids are unique per project across datasets").
    That is exactly why `generation_dataset_item_id` below exists as a second, distinct
    derivation rather than reusing this one: the two datasets share golden ids as ground
    truth (SPEC §12.5), but cannot share a Langfuse item-id namespace.
    """
    return golden_id


def generation_dataset_item_id(golden_id: str) -> str:
    """The Langfuse dataset item id for a golden-set item in the **generation** dataset —
    `dataset_item_id`'s own docstring explains why this can't just be `dataset_item_id`
    verbatim: the same bare golden id is already live in `rag-assurances-retrieval`, and
    Langfuse rejects a second dataset item claiming it project-wide. The golden id itself
    still travels through, in full, as `metadata["golden_id"]` (`generation_dataset_items`)
    — every reader that needs the *golden-set* id back (`rag.eval.run_generation_experiment`'s
    task, evaluator and per-item persistence) reads that field, never this Langfuse-specific
    id, so the prefix below is free to be whatever keeps it unique without disturbing
    anything downstream.
    """
    return f"generation-{golden_id}"


@dataclass(frozen=True)
class RetrievalDatasetItem:
    """One Langfuse dataset item's worth of content (SPEC §2.1's three-part shape: `input`,
    `expected_output`, `metadata`) — `sync_retrieval_dataset` is the only thing that turns
    this into an actual SDK call."""

    id: str
    input: str
    expected_output: dict[str, Any]
    metadata: dict[str, Any]


def retrieval_dataset_items(golden_set: Sequence[GoldenItem]) -> list[RetrievalDatasetItem]:
    """Project `rag.eval.retrieval_metrics.working_set` into dataset-item shape.

    `input` is the bare question — the working set is single-turn only (SPEC §12.1's
    `history == []` subset), so there is no history to carry. `expected_output` carries
    exactly the fields `rag.eval.retrieval_metrics.score_item` reads off a `GoldenItem`
    (`expected_state`, `gold_fiches`, `gold_articles`, `gold_spans`) — not `expected_points`,
    which belongs to the generation dataset (SPEC §12.5's other regime), and not `question`,
    which is already `input`. `id` is `dataset_item_id(item.id)` (#35 acceptance: "labels
    stay valid across every arm") so re-syncing an unchanged item updates the same dataset
    item rather than minting a duplicate.
    """
    return [
        RetrievalDatasetItem(
            id=dataset_item_id(item.id),
            input=item.question,
            expected_output={
                "expected_state": item.expected_state,
                "gold_fiches": list(item.gold_fiches),
                "gold_articles": list(item.gold_articles),
                "gold_spans": list(item.gold_spans),
            },
            metadata={"golden_id": item.id, "tags": list(item.tags)},
        )
        for item in working_set(golden_set)
    ]


def reconstruct_golden_item(*, golden_id: str, question: str, expected_output: Mapping[str, Any]) -> GoldenItem:
    """The inverse of `retrieval_dataset_items`'s projection (#35/#46's `run_experiment`
    evaluators need this: `dataset.run_experiment()` hands them a Langfuse `DatasetItem`,
    not a `GoldenItem`, and `rag.eval.retrieval_metrics.score_item` only takes the latter).

    Only reconstructs what `score_item` reads — `expected_state`, `gold_fiches`,
    `gold_articles`, `gold_spans` — plus `id`/`question` for a well-formed `GoldenItem`.
    `history` is always `()` (the working set is single-turn only, SPEC §12.1) and
    `expected_points`/`tags` are always `()`/`()` — `retrieval_dataset_items` never carries
    `expected_points` (SPEC §12.5's other, generation-dataset regime) and `score_item` never
    reads tags, so there is nothing to round-trip them from.
    """
    return GoldenItem(
        id=golden_id,
        question=question,
        history=(),
        expected_state=str(expected_output["expected_state"]),
        gold_fiches=tuple(expected_output["gold_fiches"]),
        gold_spans=tuple(expected_output["gold_spans"]),
        gold_articles=tuple(expected_output["gold_articles"]),
        expected_points=(),
        tags=(),
    )


def sync_retrieval_dataset(golden_set_path: Path, *, dataset_name: str = RETRIEVAL_DATASET_NAME) -> None:
    """Push `retrieval_dataset_items(load_golden_set(golden_set_path))` into Langfuse,
    one-directionally — the YAML is the source of truth; nothing reads back from Langfuse
    into the golden set, and an item's gold labels edited from the Langfuse UI would just be
    overwritten on the next sync (SPEC §12.5).

    No `flush()`/`shutdown()` at the end: unlike tracing (batched, needs an explicit flush in
    a short-lived script), `create_dataset_item` is a plain synchronous HTTP call
    (`dataset_items.create`), so there is nothing left buffered when this returns.
    """
    langfuse = get_client()
    try:
        langfuse.get_dataset(dataset_name)
    except NotFoundError:
        langfuse.create_dataset(name=dataset_name)

    for item in retrieval_dataset_items(load_golden_set(golden_set_path)):
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=item.id,
            input=item.input,
            expected_output=item.expected_output,
            metadata=item.metadata,
        )


@dataclass(frozen=True)
class GenerationDatasetItem:
    """One Langfuse dataset item's worth of content for the generation regime — same
    `input`/`expected_output`/`metadata` shape as `RetrievalDatasetItem`, a distinct type
    only because the two carry different fields (SPEC §12.5's "two regimes")."""

    id: str
    input: str
    expected_output: dict[str, Any]
    metadata: dict[str, Any]


def generation_dataset_items(golden_set: Sequence[GoldenItem]) -> list[GenerationDatasetItem]:
    """Project the full 60-item golden set into generation dataset-item shape — no
    `working_set` filter, unlike `retrieval_dataset_items`: generation eval scores every
    terminal state (refusals and `hors_corpus` included, SPEC §12.4's "the two empty cells
    mean opposite things") and both single- and multi-turn items (SPEC §12.1, §12.5).

    `input` is the bare current-turn question, same as retrieval's — `history` cannot live
    there too without conflating "the text to condense/generate from" with "prior turns",
    so it travels in `metadata` instead, alongside `golden_id`/`tags`. `expected_output`
    carries exactly what the generation metrics read off a `GoldenItem`: `expected_state`
    and `gold_articles` for the three deterministic ones (#46), `expected_points` for point
    coverage (#47). Not `gold_fiches`/`gold_spans` — retrieval-regime fields with no
    generation reader.
    """
    return [
        GenerationDatasetItem(
            id=generation_dataset_item_id(item.id),
            input=item.question,
            expected_output={
                "expected_state": item.expected_state,
                "gold_articles": list(item.gold_articles),
                "expected_points": list(item.expected_points),
            },
            metadata={
                "golden_id": item.id,
                "tags": list(item.tags),
                "history": [dict(turn) for turn in item.history],
            },
        )
        for item in golden_set
    ]


def reconstruct_generation_item(
    *, golden_id: str, question: str, expected_output: Mapping[str, Any], metadata: Mapping[str, Any]
) -> GoldenItem:
    """The inverse of `generation_dataset_items`'s projection — `rag.eval.run_generation_experiment`'s
    task/evaluators need this the same way `reconstruct_golden_item` serves the retrieval
    harness: `dataset.run_experiment()` hands them a Langfuse `DatasetItem`, not a
    `GoldenItem`, and both `rag.condensation.pipeline.condense` (via `history`) and
    `rag.eval.generation_metrics.score_item` (via `expected_state`/`gold_articles`) need one.

    Only reconstructs what those callers read — plus `expected_points` for the judged point
    coverage evaluator (#47). `gold_fiches`/`gold_spans` are always `()` —
    `generation_dataset_items` never carries them (they are the retrieval regime's own
    fields) — and `tags` is always `()`, same reasoning `reconstruct_golden_item` already
    gives for its own unreconstructed fields. `expected_points` is read strictly: a dataset
    synced before #47 lacks it, and a `KeyError` says "re-sync" where a silent `()` would
    read as "no points to cover" on every item.
    """
    return GoldenItem(
        id=golden_id,
        question=question,
        history=tuple(dict(turn) for turn in metadata.get("history", [])),
        expected_state=str(expected_output["expected_state"]),
        gold_fiches=(),
        gold_spans=(),
        gold_articles=tuple(expected_output["gold_articles"]),
        expected_points=tuple(expected_output["expected_points"]),
        tags=(),
    )


def sync_generation_dataset(golden_set_path: Path, *, dataset_name: str = GENERATION_DATASET_NAME) -> None:
    """Push `generation_dataset_items(load_golden_set(golden_set_path))` into Langfuse,
    one-directionally — same get-or-create-then-upsert shape as `sync_retrieval_dataset`,
    over the full golden set rather than the working set (SPEC §12.5)."""
    langfuse = get_client()
    try:
        langfuse.get_dataset(dataset_name)
    except NotFoundError:
        langfuse.create_dataset(name=dataset_name)

    for item in generation_dataset_items(load_golden_set(golden_set_path)):
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=item.id,
            input=item.input,
            expected_output=item.expected_output,
            metadata=item.metadata,
        )
