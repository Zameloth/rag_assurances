"""Golden-set id allocation — "assigned once and never reused" (SPEC §12.1, #33).

`existing_ids` alone cannot guarantee never-reused: an item annotated then later deleted
from `golden-set.yaml` would free its number for a naive `max(existing) + 1`. A small
persisted counter file is the fix — it only ever moves forward, independent of what the
golden set currently contains, so a deleted id's number is gone for good rather than
recycled on the next save.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

__all__ = ["allocate_id"]

_ID_RE = re.compile(r"^gs-(\d+)$")


def allocate_id(counter_path: Path, existing_ids: Iterable[str]) -> str:
    """Return the next `gs-<NNN>` id and advance `counter_path` past it.

    The high water mark is `max(counter file, highest existing_ids suffix)` — either one
    can be ahead: a fresh counter file bootstraps from a hand-edited golden set, and a
    counter file survives an id being deleted from the golden set after it was allocated.
    """
    from_counter = _read_counter(counter_path)
    from_existing = max((_suffix(item_id) for item_id in existing_ids), default=0)
    next_number = max(from_counter, from_existing) + 1
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    counter_path.write_text(str(next_number), encoding="utf-8")
    return f"gs-{next_number:03d}"


def _suffix(item_id: str) -> int:
    match = _ID_RE.match(item_id)
    return int(match.group(1)) if match else 0


def _read_counter(counter_path: Path) -> int:
    if not counter_path.exists():
        return 0
    text = counter_path.read_text(encoding="utf-8").strip()
    return int(text) if text else 0
