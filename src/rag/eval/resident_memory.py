"""Resident-memory measurement for rung 6's co-resident BGE-M3 + e5 arm (SPEC §14.4/§15.7,
ADR-0004, #39).

**Why this exists as its own committed fact, not just a number in a `scripts/` print.**
SPEC §15.7: "if rung 6 wins, two embedding models become co-resident at query time — that is
the single index-bearing arm the ~4.5 GB budget could veto." #41's ladder run reads this
report alongside rung 6's quality verdict rather than re-measuring, the same way it reads a
committed `eval/runs/<run-id>.json` rather than re-running a rung — a decision input has to
survive past the process that produced it.

**What is measured, and when.** Peak resident set size (`ru_maxrss`) of a process that has
both models loaded and has embedded one query-sized batch through each — the realistic
number SPEC §14.4's own accounting already reasons in ("weights only" undercounts the
Python/torch runtime and transient activations this captures for free by reading it from
the OS rather than summing parameter counts). `ru_maxrss` is a **high-water mark that never
falls** for the life of the process (`man 2 getrusage`), so it must be read from a process
that has done *query-shaped* work — one embed call per model, batch size 1, matching
`retrieve_rungN`'s own `embed([raw_turn])[0]` — never from the arm-build process, whose
500-point ingest batches (`upsert.py`'s `_UPSERT_BATCH_SIZE`) would bake a transient
bulk-embedding peak into a number meant to describe steady query-time co-residency (SPEC
§15.7's "two embedding models co-resident at query time"). `scripts/measure_e5_ram.py` is
the dedicated, build-free script that does this; nothing in this module loads either real
model, so it stays true both to `FlagEmbedding`'s ~2.3 GB-per-model cost and to unit tests
that must never pay it.
"""

from __future__ import annotations

import json
import resource
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["ResidentMemoryReport", "measure_resident_memory_mb", "write_resident_memory_report"]


@dataclass(frozen=True)
class ResidentMemoryReport:
    measured_at: str
    rss_mb: float
    models: tuple[str, ...]
    code_git_sha: str
    note: str


def measure_resident_memory_mb() -> float:
    """Peak RSS of the current process so far, in MB. `ru_maxrss` is reported in KiB on
    Linux (`man 2 getrusage`) — the only platform this project's dev containers, CI and the
    prod VPS (SPEC §14) ever run on, so there is no macOS-bytes branch to get wrong here."""
    kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kib / 1024


def write_resident_memory_report(report: ResidentMemoryReport, path: Path) -> Path:
    """Write `report` as JSON to `path`, creating parent directories as needed, and return
    the path written — `rag.eval.retrieval_run.write_run`'s own shape, for the same reason:
    machine-written, never hand-edited, JSON not YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
