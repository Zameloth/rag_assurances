"""`retrieval_dataset_items` — the pure "what gets synced" half of #35's sync script.

The actual Langfuse push (`sync_retrieval_dataset`) is paired with @Zameloth per the #35
issue comment and stays a skeleton here (see the module docstring) — nothing in this file
exercises it.
"""

from __future__ import annotations

from rag.eval.langfuse_sync import dataset_item_id, retrieval_dataset_items
from rag.eval.schema import GoldenItem


def golden_item(
    id: str = "gs-001",
    *,
    history: tuple[dict[str, str], ...] = (),
    question: str = "je suis locataire, dois-je m'assurer ?",
    expected_state: str = "reponse",
    gold_fiches: tuple[str, ...] = (),
    gold_spans: tuple[str, ...] = (),
    gold_articles: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> GoldenItem:
    return GoldenItem(
        id=id,
        question=question,
        history=history,
        expected_state=expected_state,
        gold_fiches=gold_fiches,
        gold_spans=gold_spans,
        gold_articles=gold_articles,
        expected_points=(),
        tags=tags,
    )


def test_projects_only_the_working_set() -> None:
    items = [
        golden_item("gs-001", gold_fiches=("F1",)),
        golden_item("gs-002"),  # hors_corpus-shaped: no gold contexts, excluded
        golden_item(
            "gs-003", history=({"role": "user", "content": "hi"},), gold_fiches=("F2",)
        ),  # multi-turn, excluded
    ]
    projected = retrieval_dataset_items(items)
    assert [d.id for d in projected] == [dataset_item_id("gs-001")]


def test_input_is_the_bare_question() -> None:
    items = [golden_item("gs-001", question="la question exacte", gold_fiches=("F1",))]
    [item] = retrieval_dataset_items(items)
    assert item.input == "la question exacte"


def test_expected_output_carries_every_gold_label_score_item_needs() -> None:
    items = [
        golden_item(
            "gs-001",
            expected_state="reponse",
            gold_fiches=("F1",),
            gold_articles=("CID1",),
            gold_spans=("un extrait",),
        )
    ]
    [item] = retrieval_dataset_items(items)
    assert item.expected_output == {
        "expected_state": "reponse",
        "gold_fiches": ["F1"],
        "gold_articles": ["CID1"],
        "gold_spans": ["un extrait"],
    }


def test_metadata_carries_the_golden_id_and_tags() -> None:
    items = [golden_item("gs-001", gold_fiches=("F1",), tags=("situationnel",))]
    [item] = retrieval_dataset_items(items)
    assert item.metadata == {"golden_id": "gs-001", "tags": ["situationnel"]}


def test_id_is_the_deterministic_dataset_item_id() -> None:
    items = [golden_item("gs-014", gold_fiches=("F1",))]
    [item] = retrieval_dataset_items(items)
    assert item.id == dataset_item_id("gs-014")


class TestDatasetItemId:
    def test_is_deterministic_and_derived_from_the_golden_id(self) -> None:
        assert dataset_item_id("gs-014") == dataset_item_id("gs-014")

    def test_distinguishes_different_golden_ids(self) -> None:
        assert dataset_item_id("gs-001") != dataset_item_id("gs-002")
