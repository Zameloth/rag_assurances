#!/usr/bin/env python3
"""Run the golden-set annotation helper (SPEC §12.1, #33).

**A dev tool, run by hand.** Serves a local, read-only-against-the-corpus web UI for
annotating `eval/golden/golden-set.yaml` — see `rag.eval.annotate_app` for the app itself.
No live Qdrant needed.

    uv run python scripts/annotate_golden_set.py [--port 8765] [--golden-set PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from rag.eval.annotate_app import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=None,
        help="annotate this YAML file instead of eval/golden/golden-set.yaml (e.g. the floor-calibration set, ADR-0027)",
    )
    args = parser.parse_args(argv)
    kwargs = {"golden_set_path": args.golden_set} if args.golden_set else {}
    uvicorn.run(create_app(**kwargs), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
