"""`rag.eval.paths` points into this repo, not somewhere near the installed package."""

from rag.eval.paths import CALIBRATION_SET_PATH, GOLDEN_SET_PATH, REPO_ROOT, RUNS_DIR


def test_repo_root_is_the_checkout() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "SPEC.md").is_file()


def test_committed_artifacts_are_where_the_constants_say() -> None:
    assert GOLDEN_SET_PATH.is_file()
    assert RUNS_DIR.is_dir()
    assert CALIBRATION_SET_PATH.parent.is_dir()
