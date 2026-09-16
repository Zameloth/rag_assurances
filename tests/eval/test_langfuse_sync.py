"""`retrieval_dataset_items` — the pure "what gets synced" half of #35's sync script — plus
`sync_retrieval_dataset` itself, now that it's implemented (paired with @Zameloth per the #35
issue comment).

`sync_retrieval_dataset` is exercised against a hand-written fake client
(`_FakeLangfuseClient`), monkeypatched in for `get_client`, rather than a real Langfuse
project — same posture as `test_retrieval_rerank.py`'s `monkeypatch.setattr(module, name,
fake)` pattern, and it sidesteps needing live credentials/network for a unit test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langfuse.api import NotFoundError

import rag.eval.langfuse_sync as langfuse_sync
from rag.eval.langfuse_sync import (
    RETRIEVAL_DATASET_NAME,
    dataset_item_id,
    reconstruct_golden_item,
    retrieval_dataset_items,
)
from rag.eval.schema import GoldenItem, dump_golden_set


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


class TestReconstructGoldenItem:
    def test_round_trips_every_field_score_item_reads(self) -> None:
        original = golden_item(
            "gs-014",
            expected_state="reponse",
            gold_fiches=("F1", "F2"),
            gold_articles=("CID1",),
            gold_spans=("un extrait",),
        )
        [projected] = retrieval_dataset_items([original])

        reconstructed = reconstruct_golden_item(
            golden_id=projected.id, question=projected.input, expected_output=projected.expected_output
        )

        assert reconstructed.id == original.id
        assert reconstructed.question == original.question
        assert reconstructed.expected_state == original.expected_state
        assert reconstructed.gold_fiches == original.gold_fiches
        assert reconstructed.gold_articles == original.gold_articles
        assert reconstructed.gold_spans == original.gold_spans

    def test_never_reconstructs_history_or_expected_points_or_tags(self) -> None:
        reconstructed = reconstruct_golden_item(
            golden_id="gs-001",
            question="une question",
            expected_output={
                "expected_state": "reponse",
                "gold_fiches": ["F1"],
                "gold_articles": [],
                "gold_spans": [],
            },
        )

        assert reconstructed.history == ()
        assert reconstructed.expected_points == ()
        assert reconstructed.tags == ()


class _FakeLangfuseClient:
    """Records calls instead of making them — just enough of the `Langfuse` surface for
    `sync_retrieval_dataset` to run against: `get_dataset`/`create_dataset` for the
    get-or-create branch, `create_dataset_item` for the push loop.

    `dataset_exists` stands in for "the dataset is already there" — `get_dataset` raises
    `NotFoundError` (the real SDK's own exception on a 404, per
    `rag.eval.langfuse_sync`'s module docstring) when it's `False`.
    """

    def __init__(self, *, dataset_exists: bool) -> None:
        self.dataset_exists = dataset_exists
        self.get_dataset_calls: list[str] = []
        self.create_dataset_calls: list[str] = []
        self.create_dataset_item_calls: list[dict[str, object]] = []

    def get_dataset(self, name: str) -> None:
        self.get_dataset_calls.append(name)
        if not self.dataset_exists:
            raise NotFoundError(body={"message": "not found"})

    def create_dataset(self, *, name: str) -> None:
        self.create_dataset_calls.append(name)
        self.dataset_exists = True

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        id: str,
        input: str,
        expected_output: dict[str, object],
        metadata: dict[str, object],
    ) -> None:
        self.create_dataset_item_calls.append(
            {
                "dataset_name": dataset_name,
                "id": id,
                "input": input,
                "expected_output": expected_output,
                "metadata": metadata,
            }
        )


class TestSyncRetrievalDataset:
    def test_creates_the_dataset_when_it_does_not_exist_yet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeLangfuseClient(dataset_exists=False)
        monkeypatch.setattr(langfuse_sync, "get_client", lambda: fake)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([], golden_set_path)

        langfuse_sync.sync_retrieval_dataset(golden_set_path)

        assert fake.get_dataset_calls == [RETRIEVAL_DATASET_NAME]
        assert fake.create_dataset_calls == [RETRIEVAL_DATASET_NAME]

    def test_does_not_recreate_an_existing_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeLangfuseClient(dataset_exists=True)
        monkeypatch.setattr(langfuse_sync, "get_client", lambda: fake)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([], golden_set_path)

        langfuse_sync.sync_retrieval_dataset(golden_set_path)

        assert fake.create_dataset_calls == []

    def test_pushes_one_create_dataset_item_call_per_working_set_item(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeLangfuseClient(dataset_exists=True)
        monkeypatch.setattr(langfuse_sync, "get_client", lambda: fake)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set(
            [
                golden_item("gs-001", gold_fiches=("F1",)),
                golden_item("gs-002"),  # hors_corpus-shaped: excluded from the working set
            ],
            golden_set_path,
        )

        langfuse_sync.sync_retrieval_dataset(golden_set_path)

        [call] = fake.create_dataset_item_calls
        assert call == {
            "dataset_name": RETRIEVAL_DATASET_NAME,
            "id": dataset_item_id("gs-001"),
            "input": "je suis locataire, dois-je m'assurer ?",
            "expected_output": {
                "expected_state": "reponse",
                "gold_fiches": ["F1"],
                "gold_articles": [],
                "gold_spans": [],
            },
            "metadata": {"golden_id": "gs-001", "tags": []},
        }

    def test_pushes_to_a_caller_supplied_dataset_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeLangfuseClient(dataset_exists=True)
        monkeypatch.setattr(langfuse_sync, "get_client", lambda: fake)
        golden_set_path = tmp_path / "golden-set.yaml"
        dump_golden_set([golden_item("gs-001", gold_fiches=("F1",))], golden_set_path)

        langfuse_sync.sync_retrieval_dataset(golden_set_path, dataset_name="some-other-dataset")

        assert fake.get_dataset_calls == ["some-other-dataset"]
        assert fake.create_dataset_item_calls[0]["dataset_name"] == "some-other-dataset"
