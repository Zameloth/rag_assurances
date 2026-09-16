"""One-directional sync: `eval/golden/golden-set.yaml` -> the `rag-assurances-retrieval`
Langfuse dataset (SPEC §12.5, #35).

**`retrieval_dataset_items` is plain data-shaping** — no Langfuse import, no network call —
so it is implemented and tested here like the rest of the harness. **`sync_retrieval_dataset`
was paired with @Zameloth**, per the #35 issue comment and the precedent #28/#31 already set
(`rag.retrieval.langchain_retriever`'s `_traced`).

Two calls into the SDK source (langfuse==4.15.1) settled what the docs don't say: `get_dataset`
re-raises `langfuse.api.NotFoundError` on an unknown name (so get-or-create is a plain
try/except, no need to risk `create_dataset`'s own idempotency on re-sync), and
`create_dataset_item(id=...)` "upserts if an item with id already exists" per its own
docstring — no fetch-then-update path needed. A removed golden-set item is left as a
deliberate orphan (`rag.eval.ids`'s ids are never reused, so it can never collide with a
future item) rather than archived — the simpler of the two options and a deliberate call,
not a placeholder.
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
    "RETRIEVAL_DATASET_NAME",
    "RetrievalDatasetItem",
    "dataset_item_id",
    "reconstruct_golden_item",
    "retrieval_dataset_items",
    "sync_retrieval_dataset",
]

RETRIEVAL_DATASET_NAME = "rag-assurances-retrieval"


def dataset_item_id(golden_id: str) -> str:
    """The Langfuse dataset item id for a golden-set item — golden ids are already SPEC
    §12.1's stable, hand-assigned, never-renumbered natural key, so reusing one verbatim is
    the simplest deterministic derivation and is what makes this sync idempotent (#35: "an
    id-keyed sync script"): re-running it against an unchanged item updates the same dataset
    item instead of minting a duplicate, and every arm's `run_experiment` run joins its
    `DatasetRunItem` back to the same item regardless of which rung produced it.

    If the Langfuse SDK's own item-identity rules turn out to need something other than a
    bare golden id (e.g. a UUID-shaped id), this is the one place to change — everything
    upstream only ever calls this function, never repeats the derivation.
    """
    return golden_id


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
