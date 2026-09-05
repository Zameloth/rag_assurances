"""`GoldenItem` schema round-trip and shape errors (SPEC §12.1, #33)."""

from pathlib import Path

import pytest

from rag.eval.schema import (
    EXPECTED_STATES,
    GoldenItem,
    GoldenSetSchemaError,
    dump_golden_set,
    load_golden_set,
)

ITEM = GoldenItem(
    id="gs-014",
    question="je suis locataire, je dois vraiment prendre une assurance ?",
    history=(),
    expected_state="reponse",
    gold_fiches=("F2123",),
    gold_spans=("l'assurance ... est obligatoire pour le locataire",),
    gold_articles=("LEGIARTI000006791829",),
    expected_points=("la responsabilité civile locative est obligatoire",),
    tags=("situationnel",),
)


def test_expected_states_is_the_five_envelope_shapes() -> None:
    assert {
        "reponse",
        "reponse_sans_article",
        "refus:recommandation_produit",
        "refus:conseil_action",
        "refus:hors_corpus",
    } == EXPECTED_STATES


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "golden-set.yaml"
    dump_golden_set([ITEM], path)
    assert load_golden_set(path) == [ITEM]


def test_round_trip_with_history(tmp_path: Path) -> None:
    item = GoldenItem(
        id="gs-050",
        question="et si je résilie avant la fin ?",
        history=({"role": "user", "content": "je loue un appartement"},),
        expected_state="reponse",
        gold_fiches=("F2123",),
        gold_spans=(),
        gold_articles=(),
        expected_points=("point",),
        tags=("multi_turn",),
    )
    path = tmp_path / "golden-set.yaml"
    dump_golden_set([item], path)
    assert load_golden_set(path) == [item]


def test_missing_file_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "does-not-exist.yaml"
    assert load_golden_set(path) == []


def test_empty_file_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "golden-set.yaml"
    path.write_text("")
    assert load_golden_set(path) == []


def test_missing_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden-set.yaml"
    path.write_text("- id: gs-001\n  question: a question\n")
    with pytest.raises(GoldenSetSchemaError, match="gs-001"):
        load_golden_set(path)


def test_wrong_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden-set.yaml"
    path.write_text(
        "- id: gs-001\n"
        "  question: a question\n"
        "  history: []\n"
        "  expected_state: reponse\n"
        "  gold_fiches: F2123\n"  # should be a list, not a bare string
        "  gold_spans: []\n"
        "  gold_articles: []\n"
        "  expected_points: [point]\n"
        "  tags: []\n"
    )
    with pytest.raises(GoldenSetSchemaError, match="gold_fiches"):
        load_golden_set(path)


def test_malformed_history_turn_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden-set.yaml"
    path.write_text(
        "- id: gs-001\n"
        "  question: a question\n"
        "  history: [not-a-mapping]\n"
        "  expected_state: reponse\n"
        "  gold_fiches: []\n"
        "  gold_spans: []\n"
        "  gold_articles: []\n"
        "  expected_points: [point]\n"
        "  tags: []\n"
    )
    with pytest.raises(GoldenSetSchemaError, match="history"):
        load_golden_set(path)


def test_dump_then_reload_preserves_field_order(tmp_path: Path) -> None:
    path = tmp_path / "golden-set.yaml"
    dump_golden_set([ITEM], path)
    text = path.read_text()
    fields = ["id", "question", "history", "expected_state", "gold_fiches", "gold_spans", "gold_articles", "expected_points", "tags"]
    positions = [text.index(f"{field}:") for field in fields]
    assert positions == sorted(positions)
