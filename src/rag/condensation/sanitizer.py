"""The deterministic post-schema sanitizer (SPEC §8.3, §8.4, ADR-0008, #43).

Catches what the strict `json_schema` output contract structurally cannot: preamble
wrapping the question is a schema-shaped problem (`rag.condensation.chain` fixes that by
construction), but empty output, multi-line output, a runaway length and a manufactured
article reference all still type-check as a valid one-line string. `sanitize_condensed`
is what catches those — every failure here is a caller's cue to fall back to the raw turn
(ADR-0008: "every trip falls back to the raw user turn").
"""

from __future__ import annotations

from rag.retrieval.short_circuit import scan_article_references

__all__ = ["MAX_CONDENSED_CHARS", "sanitize_condensed"]

# SPEC §8.3's table: "runaway output (> ~300 chars)". A French question restating even a
# verbose follow-up in full has no legitimate reason to run past this.
MAX_CONDENSED_CHARS = 300


def sanitize_condensed(raw_turn: str, candidate: str) -> str | None:
    """`candidate` (the schema's `requete` field) trimmed and validated, or `None` if it
    must fall back to the raw turn.

    Rejects, in SPEC §8.3's order: empty/whitespace-only, multi-line (checked after
    trimming surrounding whitespace, so a single trailing newline isn't multi-line but an
    embedded one is), runaway length, and — SPEC §8.4's reference monotonicity —
    `refs(candidate) ⊄ refs(raw_turn)`. `raw_turn` is scanned fresh on every call rather
    than passed pre-scanned: this function's whole contract is "safe against exactly this
    raw turn", so it takes the same input the short-circuit itself would have scanned.
    """
    stripped = candidate.strip()
    if not stripped:
        return None
    if "\n" in stripped:
        return None
    if len(stripped) > MAX_CONDENSED_CHARS:
        return None
    if not scan_article_references(stripped) <= scan_article_references(raw_turn):
        return None
    return stripped
