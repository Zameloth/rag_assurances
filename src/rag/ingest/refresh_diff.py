"""Corpus refresh diff (SPEC §3.3, #22).

A refresh must report three counts **before writing anything**: added, removed, and
changed-text-under-the-same-identity. The third is the dangerous one — 52% of Code
articles have been amended at least once, so a refresh is *expected* to invalidate some
hand-annotated gold labels. That is the price of freshness, not a bug; what makes it
survivable is finding out here, at refresh time, rather than as an unexplained dip in a
ladder rung three weeks later.

One function serves both registers: articles keyed by `cid` comparing `texte`, fiches
keyed by `fiche_id` comparing the verbatim XML bytes. Either shape is just an
`identity -> comparable value` mapping.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["RefreshDiff", "diff_corpus", "load_jsonl"]


@dataclass(frozen=True)
class RefreshDiff:
    added: list[str]
    removed: list[str]
    changed: list[str]

    def summary(self) -> str:
        return (
            f"added={len(self.added)} removed={len(self.removed)} "
            f"changed-text-under-the-same-id={len(self.changed)}"
        )


def diff_corpus(old: Mapping[str, object], new: Mapping[str, object]) -> RefreshDiff:
    """Compare two `identity -> comparable text` snapshots of a corpus.

    `old` empty (first ingest) reports every `new` id as added and nothing as changed —
    the correct answer, not a caller-side special case.
    """
    old_ids, new_ids = old.keys(), new.keys()
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    changed = sorted(key for key in old_ids & new_ids if old[key] != new[key])
    return RefreshDiff(added=added, removed=removed, changed=changed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a committed JSONL corpus file — one JSON object per line, as `articles.jsonl` is.

    Shared by both fetch scripts, which each need the currently-committed `articles.jsonl`
    for a different reason (fetch_fiches.py: the scope rule's join target; fetch_articles.py:
    the refresh diff's `old` side). Raises `FileNotFoundError` same as `Path.read_text` would
    — callers for whom a missing file is a legitimate first-ingest state check `path.exists()`
    themselves before calling.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]
