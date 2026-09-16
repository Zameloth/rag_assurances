"""Resident-memory measurement and its committed record — SPEC §14.4/§15.7, ADR-0004, #39.

Rung 6 is "the one index-bearing arm the ~4.5 GB budget could veto" (SPEC §15.7) because it
is the only rung with two embedding models resident at once. `measure_resident_memory_mb`
is exercised against a monkeypatched `resource.getrusage` — this file has no interest in
what BGE-M3 + e5 actually weigh, only that the KiB-to-MB conversion and the report's shape
are right; the real number is a `scripts/`-run fact, not something a unit test reproduces.
"""

from __future__ import annotations

import json
import resource
from pathlib import Path

import pytest

from rag.eval.resident_memory import (
    ResidentMemoryReport,
    measure_resident_memory_mb,
    write_resident_memory_report,
)


def test_measure_resident_memory_mb_converts_linux_kib_to_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeUsage:
        ru_maxrss = 4_500_000  # KiB, Linux's own unit for ru_maxrss

    monkeypatch.setattr(resource, "getrusage", lambda who: _FakeUsage())

    assert measure_resident_memory_mb() == pytest.approx(4_500_000 / 1024)


def test_write_resident_memory_report_creates_parent_dirs(tmp_path: Path) -> None:
    report = ResidentMemoryReport(
        measured_at="2026-09-16T12:00:00Z",
        rss_mb=4600.0,
        models=("BAAI/bge-m3", "intfloat/multilingual-e5-large-instruct"),
        code_git_sha="a" * 40,
        note="both models warmed with one embed call each before measuring",
    )
    path = tmp_path / "nested" / "rung6-resident-memory.json"

    written = write_resident_memory_report(report, path)

    assert written == path
    assert path.exists()


def test_write_resident_memory_report_round_trips_as_json(tmp_path: Path) -> None:
    report = ResidentMemoryReport(
        measured_at="2026-09-16T12:00:00Z",
        rss_mb=4600.0,
        models=("BAAI/bge-m3", "intfloat/multilingual-e5-large-instruct"),
        code_git_sha="a" * 40,
        note="both models warmed with one embed call each before measuring",
    )
    path = tmp_path / "rung6-resident-memory.json"

    write_resident_memory_report(report, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["rss_mb"] == 4600.0
    assert payload["models"] == [
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-large-instruct",
    ]
