#!/usr/bin/env python3
"""Build rung 6's e5-dense + M3-sparse arm collections (SPEC §5, ADR-0004, #39).

Builds `ARTICLES_E5_ARM`/`FICHES_E5_ARM` from the same committed corpus the incumbent arms
were built from, using `rag.ingest.e5_embedder.index_embed_batch` (e5-dense on the passage
side, M3-sparse) — no re-chunking, no alias flip (that is #41's eval-harness job, done once
a verdict is read).

Resident-memory measurement lives in `scripts/measure_e5_ram.py`, deliberately not here:
this script's own 500-point ingest batches (`upsert.py`'s `_UPSERT_BATCH_SIZE`) would bake a
transient bulk-embedding peak into a number meant to describe query-time co-residency (SPEC
§15.7) — see `rag.eval.resident_memory`'s module docstring for why the two have to be
separate processes.

    uv run python scripts/build_e5_arm.py
"""

from __future__ import annotations

import sys

from qdrant_client import QdrantClient

from rag.config import load_settings
from rag.ingest.e5_arm import (
    ARTICLES_E5_ARM,
    FICHES_E5_ARM,
    build_articles_e5_arm,
    build_fiches_e5_arm,
)


def main(argv: list[str] | None = None) -> int:
    del argv  # no options today — kept for the same CLI shape run_ab_pilot.py's main() has
    settings = load_settings()  # also loads .env into the process environment
    client = QdrantClient(settings.qdrant_url)

    # Deferred: pulls in torch, and running the corpus through it is the point (mirrors
    # run_ab_pilot.py's own deferred `embed_batch` import).
    from rag.ingest.e5_embedder import index_embed_batch

    try:
        print(f"building {ARTICLES_E5_ARM!r} ...")
        articles_written = build_articles_e5_arm(client, index_embed_batch)
        print(f"  {articles_written} point(s) written")

        print(f"building {FICHES_E5_ARM!r} ...")
        fiches_written = build_fiches_e5_arm(client, index_embed_batch)
        print(f"  {fiches_written} point(s) written")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
