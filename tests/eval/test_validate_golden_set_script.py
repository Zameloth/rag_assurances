"""`scripts/validate_golden_set.py --floor-calibration` (ADR-0027, #96), against the committed
corpus and tmp YAML files — never the real golden set or calibration file."""

import importlib.util
from pathlib import Path

import pytest

from rag.eval.schema import GoldenItem, dump_golden_set

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("validate_golden_set_script", REPO_ROOT / "scripts" / "validate_golden_set.py")
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def _item(item_id: str, fiche: str = "F1124") -> GoldenItem:
    return GoldenItem(
        id=item_id,
        question="une question ?",
        history=(),
        expected_state="reponse",
        gold_fiches=(fiche,),
        gold_spans=(),
        gold_articles=("LEGIARTI000006792738",),
        expected_points=("un point",),
        tags=(),
    )


def test_floor_calibration_flag_rejects_a_fiche_shared_with_the_golden_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    golden, calibration = tmp_path / "golden.yaml", tmp_path / "cal.yaml"
    dump_golden_set([_item("gs-001")], golden)
    dump_golden_set([_item("gs-002")], calibration)

    code = script.main(["--floor-calibration", "--golden-set", str(golden), str(calibration)])

    assert code == 1
    assert "gs-002:F1124" in capsys.readouterr().err


def test_floor_calibration_flag_accepts_a_disjoint_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    golden, calibration = tmp_path / "golden.yaml", tmp_path / "cal.yaml"
    dump_golden_set([], golden)
    dump_golden_set([_item("gs-002")], calibration)

    code = script.main(["--floor-calibration", "--golden-set", str(golden), str(calibration)])

    assert code == 0
    assert "OK" in capsys.readouterr().out
