"""Golden-set validator (SPEC §12.1-§12.4, ADR-0010, #33).

Each assertion is exercised against synthetic fiche/article corpora so the logic is
tested in isolation; `test_validate_against_the_committed_corpus` is the one integration
check that the pieces wire together against real data, the same split
`ingest/assertions.py`'s own tests make between per-assertion unit tests and a corpus run.
"""

from pathlib import Path

import pytest

from rag.eval.corpus import fiche_chunk_texts as corpus_fiche_chunk_texts
from rag.eval.schema import GoldenItem
from rag.eval.validate import (
    GoldenSetValidationError,
    validate_golden_set,
    validate_golden_set_against_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"

FICHE_IDS = {"F2123", "F9999"}
ARTICLE_CIDS = {"LEGIARTI000006791829"}
CHUNK_TEXTS = {
    "F2123": ["l'assurance responsabilité civile est obligatoire pour le locataire."],
    "F9999": ["autre fiche, autre texte, rien en commun."],
}


def _item(**overrides: object) -> GoldenItem:
    base: dict[str, object] = dict(
        id="gs-001",
        question="une question ?",
        history=(),
        expected_state="reponse",
        gold_fiches=("F2123",),
        gold_spans=("l'assurance responsabilité civile est obligatoire pour le locataire.",),
        gold_articles=("LEGIARTI000006791829",),
        expected_points=("un point",),
        tags=(),
    )
    base.update(overrides)
    return GoldenItem(**base)  # type: ignore[arg-type]


def _validate(items: list[GoldenItem]) -> None:
    validate_golden_set(
        items,
        fiche_ids=FICHE_IDS,
        article_cids=ARTICLE_CIDS,
        fiche_chunk_texts=CHUNK_TEXTS,
    )


def test_valid_item_passes() -> None:
    _validate([_item()])


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(GoldenSetValidationError, match="gs-001"):
        _validate([_item(), _item()])


def test_id_shape_rejected() -> None:
    with pytest.raises(GoldenSetValidationError, match="id"):
        _validate([_item(id="item-one")])


def test_unknown_expected_state_rejected() -> None:
    with pytest.raises(GoldenSetValidationError, match="expected_state"):
        _validate([_item(expected_state="refus")])


def test_gold_fiche_must_resolve() -> None:
    with pytest.raises(GoldenSetValidationError, match="gold_fiches"):
        _validate([_item(gold_fiches=("F0000",))])


def test_gold_article_must_resolve() -> None:
    with pytest.raises(GoldenSetValidationError, match="gold_articles"):
        _validate([_item(gold_articles=("LEGIARTI000000000000",))])


def test_gold_span_must_be_verbatim_in_a_gold_fiche_chunk() -> None:
    with pytest.raises(GoldenSetValidationError, match="gold_spans"):
        _validate([_item(gold_spans=("ce texte n'existe nulle part",))])


def test_gold_span_matching_a_different_fiches_chunk_is_still_rejected() -> None:
    # "autre fiche" text exists, but not among *this* item's gold_fiches (F2123).
    with pytest.raises(GoldenSetValidationError, match="gold_spans"):
        _validate([_item(gold_spans=("autre fiche, autre texte, rien en commun.",))])


def test_reponse_sans_article_requires_empty_articles_and_nonempty_fiches() -> None:
    _validate(
        [
            _item(
                id="gs-002",
                expected_state="reponse_sans_article",
                gold_articles=(),
                gold_spans=(),
            )
        ]
    )


def test_reponse_sans_article_with_populated_articles_rejected() -> None:
    with pytest.raises(GoldenSetValidationError, match="reponse_sans_article"):
        _validate([_item(id="gs-002", expected_state="reponse_sans_article")])


def test_hors_corpus_requires_all_empty() -> None:
    _validate(
        [
            _item(
                id="gs-003",
                expected_state="refus:hors_corpus",
                gold_fiches=(),
                gold_articles=(),
                gold_spans=(),
                expected_points=(),
            )
        ]
    )


def test_hors_corpus_with_populated_expected_points_rejected() -> None:
    """SPEC §12.9 — "refusal items carry points too, ... hors_corpus items carry none"."""
    with pytest.raises(GoldenSetValidationError, match="expected_points"):
        _validate(
            [
                _item(
                    id="gs-003",
                    expected_state="refus:hors_corpus",
                    gold_fiches=(),
                    gold_articles=(),
                    gold_spans=(),
                )
            ]
        )


def test_hors_corpus_with_a_gold_fiche_rejected() -> None:
    with pytest.raises(GoldenSetValidationError, match="hors_corpus"):
        _validate(
            [
                _item(
                    id="gs-003",
                    expected_state="refus:hors_corpus",
                    gold_articles=(),
                )
            ]
        )


def test_reponse_requires_nonempty_fiches_and_articles() -> None:
    with pytest.raises(GoldenSetValidationError, match="gold_fiches"):
        _validate([_item(gold_fiches=(), gold_spans=())])


def test_expected_points_must_have_one_to_three_entries() -> None:
    with pytest.raises(GoldenSetValidationError, match="expected_points"):
        _validate([_item(expected_points=())])
    with pytest.raises(GoldenSetValidationError, match="expected_points"):
        _validate([_item(expected_points=("a", "b", "c", "d"))])


def test_every_violation_is_reported_together() -> None:
    bad = _item(id="item-one", expected_state="refus", gold_fiches=("F0000",))
    with pytest.raises(GoldenSetValidationError) as excinfo:
        _validate([bad])
    message = str(excinfo.value)
    assert "id" in message
    assert "expected_state" in message
    assert "gold_fiches" in message


def test_validate_against_the_committed_corpus() -> None:
    """Wires `validate_golden_set` to real fiche/article resolution and real chunk texts —
    the F1124 fixture already exercised in `test_eval_corpus.py`."""
    span = next(t for t in corpus_fiche_chunk_texts("F1124", FICHES_DIR) if "association" in t.lower())
    item = _item(
        id="gs-100",
        gold_fiches=("F1124",),
        gold_spans=(span[:40],),
        gold_articles=("LEGIARTI000006792738",),  # L127-1, under F1124's dc:source section
    )
    validate_golden_set_against_corpus([item], fiches_dir=FICHES_DIR, articles_path=ARTICLES_PATH)


def test_validate_against_the_committed_corpus_rejects_a_bad_cid() -> None:
    item = _item(id="gs-101", gold_fiches=("F1124",), gold_spans=(), gold_articles=("LEGIARTI000000000000",))
    with pytest.raises(GoldenSetValidationError, match="gold_articles"):
        validate_golden_set_against_corpus([item], fiches_dir=FICHES_DIR, articles_path=ARTICLES_PATH)
