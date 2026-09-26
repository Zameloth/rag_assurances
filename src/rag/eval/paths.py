"""Where the eval's committed artifacts live in the repo (SPEC §12.11), for the scripts
that read and write them.

Resolved from this file's location, so it holds for the editable install `uv sync` makes —
the only way these dev scripts are run; a built wheel has no repo around it to point into.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["CALIBRATION_SET_PATH", "GOLDEN_SET_PATH", "REPO_ROOT", "RUNS_DIR"]

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_SET_PATH = REPO_ROOT / "eval" / "golden" / "golden-set.yaml"
# SPEC §12.11 names the file `judge-set.yaml`; the glossary term for its contents is
# "the calibration set" (CONTEXT.md).
CALIBRATION_SET_PATH = REPO_ROOT / "eval" / "calibration" / "judge-set.yaml"
RUNS_DIR = REPO_ROOT / "eval" / "runs"
