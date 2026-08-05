#!/usr/bin/env python3
"""Run the golden-set annotation helper (SPEC §12.1, #33).

**A dev tool, run by hand.** Serves a local, read-only-against-the-corpus web UI for
annotating `eval/golden/golden-set.yaml` — see `rag.eval.annotate_app` for the app itself.
No live Qdrant needed.

    uv run python scripts/annotate_golden_set.py [--port 8765]
"""

from __future__ import annotations

import argparse

import uvicorn

from rag.eval.annotate_app import app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
