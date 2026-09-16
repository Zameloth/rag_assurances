#!/usr/bin/env python3
"""Measure resident memory with BGE-M3 and e5 both loaded, query-time-shaped (SPEC §14.4,
§15.7, ADR-0004, #39).

Deliberately does **not** build any collection or touch Qdrant. `rag.eval.resident_memory`'s
module docstring explains why: `ru_maxrss` is a high-water mark that never falls for the
life of a process, so it has to be read from a process that only ever did *query-shaped*
work — one embed call per model, batch size 1, matching `retrieve_rungN`'s own
`embed([raw_turn])[0]` — never from `scripts/build_e5_arm.py`'s 500-point ingest batches,
which would bake a transient bulk-embedding peak into a number meant to describe SPEC
§15.7's "two embedding models co-resident at query time."

The measurement is written to `eval/runs/rung6-resident-memory.json` so #41's ladder run can
cite it next to rung 6's quality verdict instead of re-measuring.

    uv run python scripts/measure_e5_ram.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.eval.resident_memory import (
    ResidentMemoryReport,
    measure_resident_memory_mb,
    write_resident_memory_report,
)
from rag.eval.retrieval_run import resolve_git_sha

REPO_ROOT = Path(__file__).resolve().parents[1]
RESIDENT_MEMORY_PATH = REPO_ROOT / "eval" / "runs" / "rung6-resident-memory.json"

# A representative single question, the same shape every `retrieve_rungN` call embeds —
# batch size 1, not a bulk-ingest batch.
_WARMUP_QUERY = "Quel est le délai de résiliation d'un contrat d'assurance habitation ?"


def main(argv: list[str] | None = None) -> int:
    del argv

    # Deferred: pulls in torch, and loading both models is the point.
    from rag.ingest.e5_embedder import MODEL_ID as E5_MODEL_ID
    from rag.ingest.e5_embedder import query_embed_batch
    from rag.ingest.embedder import MODEL_ID as M3_MODEL_ID

    query_embed_batch([_WARMUP_QUERY])  # loads and runs both BGE-M3 and e5, once each

    rss_mb = measure_resident_memory_mb()
    report = ResidentMemoryReport(
        measured_at=datetime.now(UTC).isoformat(),
        rss_mb=rss_mb,
        models=(M3_MODEL_ID, E5_MODEL_ID),
        code_git_sha=resolve_git_sha(REPO_ROOT),
        note=(
            "peak RSS (ru_maxrss) after one query-sized (batch-of-1) embed call through "
            "rag.ingest.e5_embedder.query_embed_batch, i.e. with BGE-M3 and "
            "e5-large-instruct both resident and warmed the way a query-serving process "
            "would be — SPEC §15.7's rung-6-wins worst case for the ~4.5 GB prod budget"
        ),
    )
    path = write_resident_memory_report(report, RESIDENT_MEMORY_PATH)
    print(f"resident memory: {rss_mb:.0f} MB with both embedders loaded")
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
