"""Sentence-then-word band-fitting fallback, shared by the article and fiche chunkers (SPEC
§4.1 step 3, §4.2, ADR-0003) — the last-resort split for a unit that still exceeds the band
after its own structural boundaries (`texteHtml` blocks for articles, XML descent for
fiches) are exhausted. Same algorithm, same rarity, in both registers: measured at ~2 blocks
across the whole article corpus, so this is a safety valve, not the common path.
"""

from __future__ import annotations

import re

from rag.ingest.tokenizer import SPECIAL_TOKEN_COUNT, count_content_tokens

__all__ = ["split_to_band"]

# Splits after sentence-ending punctuation followed by whitespace and what looks like the
# next sentence's start. False splits inside a legal cross-reference like "l'article L. 344-2"
# just hand the caller smaller units to re-merge, not lost content.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.?!])\s+(?=[A-ZÀ-Ü0-9])")


def split_to_band(text: str, band: int) -> list[str]:
    """Sentence-split `text`, falling back to a word-group split for any sentence that still
    overflows `band` on its own — not observed on the current corpus, kept so a pathological
    future block can never silently break the "no chunk over band" assertion."""
    sentences = [s for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
    units: list[str] = []
    for sentence in sentences:
        if count_content_tokens(sentence) + SPECIAL_TOKEN_COUNT > band:
            units.extend(_split_by_words(sentence, band))
        else:
            units.append(sentence)
    return units


def _split_by_words(text: str, band: int) -> list[str]:
    words = text.split()
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = count_content_tokens(word)
        if current and current_tokens + word_tokens + SPECIAL_TOKEN_COUNT > band:
            groups.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        groups.append(" ".join(current))
    return groups
