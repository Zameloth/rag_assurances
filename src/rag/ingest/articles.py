"""Article chunking under the band (SPEC §4.2, ADR-0003, #23).

One article is one point where its table-stripped text fits the 512-token band. Otherwise
it splits on `texteHtml` blocks (`<p>`/`<br>`/`<li>`/`<tr>`) packed greedily to the band, with
a sentence-split fallback for the rare block that doesn't fit the band on its own — measured
at exactly 2 blocks across the whole corpus, so the fallback is a safety valve, not the
common path.

Where the resulting prose falls under the 32-token floor, the article becomes a stub point:
its `citation_id`, the section it sits under and a Légifrance URL are appended so the point
still clears assertion 8 (SPEC §4.5) and stays citable. Two populations land here — 7
annexes whose entire body was a `<table>` (stripped to near-nothing, so a `Table non
indexée` marker replaces the missing prose) and a larger set of genuinely short in-force
articles with no table at all (e.g. L111-4, 21 tokens of real prose): those keep their
verbatim text and are only *padded* with the citation suffix, never overwritten, because
there is nothing to disclose a table stripped and no reason to discard real content.

Payload assembly (SPEC §7.2 field for field, point ids, collections) is #25's job — this
module only turns one article row into its ordered `text` chunks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rag.ingest.html_blocks import extract_blocks, extract_text
from rag.ingest.text_split import split_to_band
from rag.ingest.tokenizer import SPECIAL_TOKEN_COUNT, count_content_tokens, count_tokens

__all__ = ["ArticleChunk", "ArticleRow", "BAND", "STUB_FLOOR", "chunk_article"]

ArticleRow = Mapping[str, Any]

BAND = 512
STUB_FLOOR = 32

_LEGIFRANCE_URL = "https://www.legifrance.gouv.fr/codes/article_lc/{legiarti_version_id}"
# Lowercase, verbatim from the ticket and SPEC §4.2: "a `table non indexée` marker".
_STUB_MARKER = "table non indexée"


@dataclass(frozen=True)
class ArticleChunk:
    text: str
    chunk_index: int
    tokens: int
    is_stub: bool


def chunk_article(row: ArticleRow) -> list[ArticleChunk]:
    """Chunk one `articles.jsonl` row into its ordered points.

    Always returns at least one chunk (assertion 6) — the stub path is what keeps that true
    for an annexe whose entire body is a stripped `<table>`.
    """
    whole_text = extract_text(row["texteHtml"])
    whole_tokens = count_tokens(whole_text)

    if whole_tokens <= BAND:
        if whole_tokens < STUB_FLOOR:
            return [_stub_chunk(row, whole_text)]
        return [ArticleChunk(text=whole_text, chunk_index=0, tokens=whole_tokens, is_stub=False)]

    packed = _pack(_units(row["texteHtml"]))
    return [
        ArticleChunk(text=text, chunk_index=index, tokens=count_tokens(text), is_stub=False)
        for index, text in enumerate(packed)
    ]


def _stub_chunk(row: ArticleRow, prose: str) -> ArticleChunk:
    """`citation_id` + title + [a table marker, only where a `<table>` was actually stripped]
    + the Légifrance URL, appended to whatever real prose survived — never in place of it."""
    url = _LEGIFRANCE_URL.format(legiarti_version_id=row["id"])
    title = (row.get("sectionParentTitre") or row["citation_id"]).rstrip(".")
    marker = f" {_STUB_MARKER}." if "<table" in row["texteHtml"] else ""
    suffix = f"{row['citation_id']} — {title}.{marker} Texte intégral : {url}"
    text = f"{prose} {suffix}" if prose else suffix
    return ArticleChunk(text=text, chunk_index=0, tokens=count_tokens(text), is_stub=True)


def _units(html: str) -> list[str]:
    """`texteHtml` blocks, with any block that overflows the band on its own pre-split into
    sentences — so `_pack` only ever sees units that individually fit."""
    units: list[str] = []
    for block in extract_blocks(html):
        if count_content_tokens(block) + SPECIAL_TOKEN_COUNT > BAND:
            units.extend(split_to_band(block, BAND))
        else:
            units.append(block)
    return units


def _pack(units: list[str]) -> list[str]:
    """Greedy bin-packing to the band, in document order — the source of the "no overlap"
    property (SPEC §4.3): every cut lands on a unit boundary a DILA editor already put there.

    A pure forward greedy pass can strand a small last unit behind a chunk that is already
    full — assertion 8 (SPEC §4.5) forbids that, so `_rebalance_tail` shifts units across the
    last boundary until the tail clears the floor, without changing the chunk count.
    """
    groups = _greedy_groups(units)
    groups = _rebalance_tail(groups)
    return [" ".join(group) for group in groups]


def _greedy_groups(units: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = count_content_tokens(unit)
        if current and current_tokens + unit_tokens + SPECIAL_TOKEN_COUNT > BAND:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        groups.append(current)
    return groups


def _group_tokens(group: list[str]) -> int:
    return sum(count_content_tokens(unit) for unit in group) + SPECIAL_TOKEN_COUNT


def _rebalance_tail(groups: list[list[str]]) -> list[list[str]]:
    """If packing left a sub-floor final chunk, borrow units from the end of the previous
    chunk until the tail clears the floor — merging outright where that fits the band, or
    shifting one unit at a time where it doesn't (LEGIARTI000006798341 is the corpus's one
    live case: a 501-token chunk followed by a lone 17-token block)."""
    if len(groups) < 2 or _group_tokens(groups[-1]) >= STUB_FLOOR:
        return groups

    previous, tail = list(groups[-2]), list(groups[-1])
    if _group_tokens(previous + tail) <= BAND:
        return [*groups[:-2], previous + tail]

    while _group_tokens(tail) < STUB_FLOOR and len(previous) > 1:
        tail.insert(0, previous.pop())
    return [*groups[:-2], previous, tail]
