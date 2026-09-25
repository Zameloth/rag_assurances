#!/usr/bin/env python3
"""Measure resident memory with BGE-M3 and the rung-4 reranker both loaded, query-time-
shaped (SPEC §14.4, §17.2, ADR-0018, #41).

**Why this exists as its own committed fact, not just a number in a `scripts/` print.**
SPEC §14.4's pre-registered rule ("prod runs the ladder-winning arm if it fits a ~4.5 GB
budget... never discovered at deploy time") and §17.2 ("there is nothing left to
decide... it needs numbers the ladder has not produced") both name a real measurement as
the missing input — SPEC §14.4 itself only ever gives an *estimate* ("Realistic peak with
both models: ~5.2-5.5 GB"), never a measured one. This script produces that number the same
way `scripts/measure_e5_ram.py` produced rung 6's (ADR-0022, #39): read alongside rung 4's
quality verdict in #41, never re-measured by it.

**What is measured, and when.** Peak resident set size (`ru_maxrss`) of a process that has
loaded BGE-M3 (the retriever's own embedder — every real query embeds through it regardless
of which rung is active) and `bge-reranker-v2-m3` (SPEC §9.4's default rung-4 arm), and has
done one query-shaped call through each — one `embed_batch` call and one `rerank` call, both
batch size 1, matching `retrieve_rung4`'s own `embed([raw_turn])[0]` and its single
`rerank(raw_turn, merged)` call. `ru_maxrss` never falls for the life of a process (`man 2
getrusage`), so this deliberately never touches Qdrant or runs a bulk build — only the
steady query-time co-residency SPEC §14.4 actually asks about.

The measurement is written to `eval/runs/rung4-resident-memory.json` so #41's ladder run can
cite it next to rung 4's quality verdict instead of re-measuring.

    uv run python scripts/measure_reranker_ram.py
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
RESIDENT_MEMORY_PATH = REPO_ROOT / "eval" / "runs" / "rung4-resident-memory.json"

# The same representative question `scripts/measure_e5_ram.py` uses — a query-shaped batch
# of one, not a bulk-ingest batch.
_WARMUP_QUERY = "Quel est le délai de résiliation d'un contrat d'assurance habitation ?"


def main(argv: list[str] | None = None) -> int:
    del argv

    # Deferred: pulls in torch, and loading both models is the point.
    from rag.ingest.embedder import MODEL_ID as M3_MODEL_ID
    from rag.ingest.embedder import embed_batch
    from rag.retrieval.candidates import Candidate, Provenance, Register
    from rag.retrieval.rerank import DEFAULT_RERANKER_MODEL, rerank

    embed_batch([_WARMUP_QUERY])  # loads and runs BGE-M3, once — the dense/sparse legs' own cost

    warmup_candidate = Candidate(
        id="warmup",
        score=0.0,
        register=Register.ARTICLE,
        payload={"text": "Le contrat peut être résilié chaque année moyennant un préavis."},
        provenance=frozenset({Provenance.SEARCH}),
    )
    rerank(_WARMUP_QUERY, [warmup_candidate])  # loads and runs bge-reranker-v2-m3, once

    rss_mb = measure_resident_memory_mb()
    report = ResidentMemoryReport(
        measured_at=datetime.now(UTC).isoformat(),
        rss_mb=rss_mb,
        models=(M3_MODEL_ID, DEFAULT_RERANKER_MODEL),
        code_git_sha=resolve_git_sha(REPO_ROOT),
        note=(
            "peak RSS (ru_maxrss) after one query-sized (batch-of-1) embed call through "
            "rag.ingest.embedder.embed_batch and one query-sized rerank call through "
            "rag.retrieval.rerank.rerank (fp32 backend, SPEC §9.4's default rung-4 arm), "
            "i.e. with BGE-M3 and bge-reranker-v2-m3 both resident and warmed the way a "
            "query-serving process would be — SPEC §14.4/§17.2's missing measured number "
            "for the ~4.5 GB prod budget rule"
        ),
    )
    path = write_resident_memory_report(report, RESIDENT_MEMORY_PATH)
    print(f"resident memory: {rss_mb:.0f} MB with BGE-M3 + reranker (fp32) loaded")
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
