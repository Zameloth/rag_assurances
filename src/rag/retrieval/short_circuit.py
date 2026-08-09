"""The article-reference short-circuit and membership check (SPEC §9.1, #27).

A query carrying an article reference skips search entirely and resolves against
`lookup_key`. This exists because of measured tokenizer behaviour: under XLM-R, `L113-2`
and `L. 113-2` — the same citation — tokenize to different subword pieces, and no sparse
weighting fixes that. The article number is a metadata field, so it doesn't need to.

**One normalizer, two patterns.** `rag.ingest.lookup_key.normalize_lookup_key` is the
anchored field validator — it validates a whole `lookup_key` value and never matches a
reference embedded in a sentence. This module's `scan_article_reference` is its unanchored
counterpart: same character shape (`LOOKUP_KEY_BODY`), but matched anywhere in free text
with a maximal-munch guarantee, so `L132-9-3-1` scans whole rather than truncating to the
real-but-shorter `L132-9-3`.

**The scan reads the raw user turn only, never the condensed query (§8.4).** Condensation
is a separate, later stage — this module is what tells that stage whether it runs at all.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from rag.ingest.lookup_key import LOOKUP_KEY_BODY, normalize_lookup_key

__all__ = [
    "ShortCircuitPath",
    "ShortCircuitResult",
    "resolve_short_circuit",
    "scan_article_reference",
]

# Unanchored: the same body as the field validator, word-boundary-terminated instead of
# `^...$`-anchored. `(?:-\d+)*` is greedy by default, which *is* the maximal-munch
# guarantee — at a given start position it consumes every dash segment it can before the
# trailing `\b` is checked, so `L113-15-2` can never be handed back as `L113-15`.
_SCAN_PATTERN = re.compile(rf"\b{LOOKUP_KEY_BODY}\b")


class ShortCircuitPath(enum.Enum):
    """SPEC §9.1's three paths — computed by code, never inferred by a caller."""

    RESOLVED = "resolved"  # path 1: reference in raw turn, membership passes
    MEMBERSHIP_FAILED = "membership_failed"  # path 2: reference in raw turn, membership fails
    NO_REFERENCE = "no_reference"  # path 3: no reference in the raw turn


@dataclass(frozen=True)
class ShortCircuitResult:
    """`path` decides the branch; `lookup_key` is set only when `path` is `RESOLVED`."""

    path: ShortCircuitPath
    lookup_key: str | None = None


def scan_article_reference(raw_turn: str) -> str | None:
    """Return the first article reference found in `raw_turn`, or `None`.

    Unanchored maximal-munch scan (SPEC §9.1) — the raw matched span, not yet normalized.
    Matches every measured `num` shape (plain, multi-segment, asterisk, no-dash) embedded
    anywhere in free text; matches nothing when no such span is present.
    """
    match = _SCAN_PATTERN.search(raw_turn)
    return match.group() if match else None


def resolve_short_circuit(raw_turn: str, lookup_keys: AbstractSet[str]) -> ShortCircuitResult:
    """Decide which of SPEC §9.1's three paths `raw_turn` takes.

    Scans `raw_turn` for an article reference, then checks the normalized match against
    `lookup_keys` — the collection's loaded `lookup_key` set. A failed membership check
    falls through exactly like no reference at all: this converts any future regex gap into
    normal retrieval rather than a confidently wrong article.
    """
    candidate = scan_article_reference(raw_turn)
    if candidate is None:
        return ShortCircuitResult(path=ShortCircuitPath.NO_REFERENCE)

    key = normalize_lookup_key(candidate)
    if key is not None and key in lookup_keys:
        return ShortCircuitResult(path=ShortCircuitPath.RESOLVED, lookup_key=key)
    return ShortCircuitResult(path=ShortCircuitPath.MEMBERSHIP_FAILED)
