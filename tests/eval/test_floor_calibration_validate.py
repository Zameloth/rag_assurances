"""Floor-calibration set validator (ADR-0027, #96).

`eval/golden/floor-calibration.yaml` uses the golden schema and the golden checks, plus two
rules of its own: single-turn only, and no `gold_fiches` shared with the golden set (which
must stay an untouched held-out test). Fixtures only — the real file is written by #95.
"""

import pytest

from rag.eval.schema import GoldenItem
from rag.eval.validate import GoldenSetValidationError, validate_floor_calibration_set

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


def _validate(items: list[GoldenItem], golden: list[GoldenItem]) -> None:
    validate_floor_calibration_set(
        items,
        golden_items=golden,
        fiche_ids=FICHE_IDS,
        article_cids=ARTICLE_CIDS,
        fiche_chunk_texts=CHUNK_TEXTS,
    )


def test_set_with_distinct_fiches_passes() -> None:
    golden = [_item(id="gs-002", gold_fiches=("F9999",), gold_spans=())]

    _validate([_item()], golden)


def test_rejects_multi_turn_items() -> None:
    multi_turn = _item(history=({"role": "user", "content": "bonjour"},), tags=("multi_turn",))

    with pytest.raises(GoldenSetValidationError, match=r"single-turn.*gs-001"):
        _validate([multi_turn], golden=[])


def test_rejects_gold_fiche_overlap_with_the_golden_set() -> None:
    golden = [_item(id="gs-007", gold_fiches=("F2123", "F9999"), gold_spans=())]

    with pytest.raises(GoldenSetValidationError, match=r"gs-001:F2123"):
        _validate([_item()], golden)


def test_still_runs_the_golden_schema_checks() -> None:
    with pytest.raises(GoldenSetValidationError, match="expected_state"):
        _validate([_item(expected_state="nonsense")], golden=[])


def test_reports_every_violation_together() -> None:
    multi_turn = _item(history=({"role": "user", "content": "bonjour"},), tags=("multi_turn",))
    golden = [_item(id="gs-007", gold_spans=())]

    with pytest.raises(GoldenSetValidationError) as excinfo:
        _validate([multi_turn], golden)

    assert "single-turn" in str(excinfo.value)
    assert "gs-001:F2123" in str(excinfo.value)
