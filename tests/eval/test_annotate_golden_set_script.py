"""`scripts/annotate_golden_set.py --golden-set` (#96): the human validation step of the
floor-calibration set annotates an alternate YAML file through the same helper."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("annotate_golden_set_script", REPO_ROOT / "scripts" / "annotate_golden_set.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_golden_set_flag_reaches_create_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = _load_script()
    created: dict[str, Any] = {}
    sentinel_app = object()

    def fake_create_app(**kwargs: Any) -> object:
        created.update(kwargs)
        return sentinel_app

    served: dict[str, Any] = {}
    monkeypatch.setattr(script, "create_app", fake_create_app)
    monkeypatch.setattr(script.uvicorn, "run", lambda app, **kw: served.update(app=app, **kw))
    alternate = tmp_path / "floor-calibration.yaml"

    script.main(["--golden-set", str(alternate), "--port", "9001"])

    assert created["golden_set_path"] == alternate
    assert served["app"] is sentinel_app
    assert served["port"] == 9001


def test_default_keeps_the_real_golden_set(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_script()
    created: dict[str, Any] = {}
    monkeypatch.setattr(script, "create_app", lambda **kwargs: created.update(kwargs) or object())
    monkeypatch.setattr(script.uvicorn, "run", lambda app, **kw: None)

    script.main([])

    assert "golden_set_path" not in created or created["golden_set_path"] == REPO_ROOT / "eval" / "golden" / "golden-set.yaml"
