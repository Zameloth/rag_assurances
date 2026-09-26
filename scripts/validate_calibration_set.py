#!/usr/bin/env python3
"""Validate `eval/calibration/judge-set.yaml` against SPEC §12.10 and the golden set (#47).

    uv run python scripts/validate_calibration_set.py [path/to/judge-set.yaml]

Exits non-zero and prints every violation found — schema shape first, then the per-pair
checks (one pair per golden item, question/points matching it, each archetype's structural
rules) — and prints the pair count per archetype otherwise.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from rag.eval.calibration import CalibrationSetError, load_calibration_set, validate_calibration_set
from rag.eval.paths import CALIBRATION_SET_PATH, GOLDEN_SET_PATH
from rag.eval.schema import load_golden_set


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else CALIBRATION_SET_PATH
    try:
        pairs = load_calibration_set(path)
    except CalibrationSetError as exc:
        print(f"{path}: INVALID\n{exc}", file=sys.stderr)
        return 1
    violations = validate_calibration_set(pairs, load_golden_set(GOLDEN_SET_PATH))
    if violations:
        print(f"{path}: INVALID\n" + "\n".join(violations), file=sys.stderr)
        return 1
    by_archetype = Counter(pair.archetype.value for pair in pairs)
    print(f"{path}: OK — {len(pairs)} pair(s)")
    for archetype, count in sorted(by_archetype.items()):
        print(f"  {archetype}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
