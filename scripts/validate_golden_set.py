#!/usr/bin/env python3
"""Validate `eval/golden/golden-set.yaml` against SPEC §12.1-§12.4 (#33).

**A dev tool, run by hand** — the same posture as `fetch_articles.py`/`fetch_fiches.py`,
but with no fetch group to activate: `rag.eval` ships in the default dependency set since
`validate_golden_set_against_corpus` is also called from the annotation helper's save path
(`rag.eval.annotate_app`), not only from here.

    uv run python scripts/validate_golden_set.py [path/to/golden-set.yaml]

    uv run python scripts/validate_golden_set.py --floor-calibration [path/to/floor-calibration.yaml]

Exits non-zero and prints every violation found (schema shape first, then the corpus-level
checks) if the file is invalid; prints the item count and exits zero otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.eval.schema import GoldenSetSchemaError, load_golden_set
from rag.eval.validate import (
    GoldenSetValidationError,
    validate_floor_calibration_set_against_corpus,
    validate_golden_set_against_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_SET = REPO_ROOT / "eval" / "golden" / "golden-set.yaml"
DEFAULT_FLOOR_CALIBRATION_SET = REPO_ROOT / "eval" / "golden" / "floor-calibration.yaml"
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="YAML file to validate (default: the golden set)")
    parser.add_argument(
        "--floor-calibration",
        action="store_true",
        help="validate a floor-calibration set (ADR-0027): single-turn only, no gold_fiches shared "
        "with the golden set. PATH defaults to eval/golden/floor-calibration.yaml",
    )
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET, help="golden set to check overlap against")
    args = parser.parse_args(argv)

    path = args.path or (DEFAULT_FLOOR_CALIBRATION_SET if args.floor_calibration else DEFAULT_GOLDEN_SET)
    try:
        items = load_golden_set(path)
        if args.floor_calibration:
            validate_floor_calibration_set_against_corpus(
                items,
                golden_items=load_golden_set(args.golden_set),
                fiches_dir=FICHES_DIR,
                articles_path=ARTICLES_PATH,
            )
        else:
            validate_golden_set_against_corpus(items, fiches_dir=FICHES_DIR, articles_path=ARTICLES_PATH)
    except (GoldenSetSchemaError, GoldenSetValidationError) as exc:
        print(f"{path}: INVALID\n{exc}", file=sys.stderr)
        return 1
    print(f"{path}: OK — {len(items)} item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
