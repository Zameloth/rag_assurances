"""Ingest assertions 1–9 (SPEC §7.4, §4.5).

These gate the corpus, not only a later ingest run: the fetch scripts (dev tools, SPEC
§3.3) call `run_article_assertions` / `run_fiche_assertions` before they write, and the
test suite calls them again against the committed files so a corpus that fails its own
assertions can never land on `main`. `run_article_chunk_assertions` (6–9, #23) gates the
*chunker* rather than the raw corpus file, so it is run separately, against `chunk_article`
output rather than `articles.jsonl` rows.

Every assertion raises `CorpusAssertionError` with every violation it found, not just the
first — a refresh that fails for three unrelated reasons should say so once, not three
runs apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rag.ingest.articles import BAND, STUB_FLOOR, ArticleChunk, ArticleRow, chunk_article
from rag.ingest.html_blocks import extract_text
from rag.ingest.lookup_key import normalize_lookup_key
from rag.ingest.tokenizer import SPECIAL_TOKEN_COUNT, count_tokens

__all__ = [
    "CorpusAssertionError",
    "run_article_assertions",
    "run_article_chunk_assertions",
    "run_fiche_assertions",
]

FicheRow = Mapping[str, Any]

# One (row, its chunks) pair per article — threaded through all four assertion-6-9 helpers
# below, so it earns a name rather than repeating the tuple-of-list shape at each signature.
_Chunked = list[tuple[ArticleRow, list[ArticleChunk]]]

# Generous upper bound on what a stub's citation_id + title + Légifrance URL suffix can add
# (measured max on the committed corpus: 98 tokens for the whole stub chunk) — wide enough
# to absorb a longer title after a refresh without chasing a tight constant.
_STUB_PADDING_BUDGET = 200


class CorpusAssertionError(Exception):
    """One or more ingest assertions failed against the corpus."""


def run_article_assertions(rows: Iterable[ArticleRow]) -> None:
    """Run assertions 1–4 over `rows`, raising with every violation found.

    `rows` is consumed once — callers passing a generator should list() it first if they
    need it again afterwards.
    """
    materialized = list(rows)
    violations: list[str] = [
        *_assert_one_version_per_chronicle(materialized),
        *_assert_all_in_force(materialized),
        *_assert_lookup_keys_valid(materialized),
        *_assert_citation_ids_present(materialized),
    ]
    if violations:
        raise CorpusAssertionError(
            f"{len(violations)} ingest assertion violation(s):\n" + "\n".join(violations)
        )


def _assert_one_version_per_chronicle(rows: list[ArticleRow]) -> list[str]:
    """Assertion 1 — `len(set(cid)) == len(rows)`; a collision means `etat` leaked."""
    cids = [row["cid"] for row in rows]
    if len(set(cids)) == len(cids):
        return []
    duplicates = sorted({cid for cid in cids if cids.count(cid) > 1})
    return [f"assertion 1: {len(duplicates)} cid(s) appear more than once: {duplicates}"]


def _assert_all_in_force(rows: list[ArticleRow]) -> list[str]:
    """Assertion 2 — every `etat == 'VIGUEUR'`."""
    offenders = sorted({row["cid"] for row in rows if row["etat"] != "VIGUEUR"})
    if not offenders:
        return []
    return [f"assertion 2: {len(offenders)} row(s) not etat=VIGUEUR: {offenders}"]


def _assert_lookup_keys_valid(rows: list[ArticleRow]) -> list[str]:
    """Assertion 3 — every non-null `lookup_key` matches the strict pattern and is unique.

    `articles.jsonl` does not carry a `lookup_key` column — SPEC §7.2 places it at the
    chunk-payload level, computed at ingest time. Here it is derived from `citation_id`
    for the duration of the check: pattern-validity is guaranteed by construction (the
    normalizer never returns a string that doesn't match), so the substantive question is
    whether two distinct articles' `citation_id`s normalize to the same key.
    """
    keys = [
        key
        for row in rows
        if row.get("citation_id")
        and (key := normalize_lookup_key(row["citation_id"])) is not None
    ]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if not duplicates:
        return []
    return [f"assertion 3: {len(duplicates)} duplicate lookup_key value(s): {duplicates}"]


def _assert_citation_ids_present(rows: list[ArticleRow]) -> list[str]:
    """Assertion 4 — every article has a non-null `citation_id`."""
    offenders = sorted(
        {row["cid"] for row in rows if not row.get("citation_id")},
    )
    if not offenders:
        return []
    return [f"assertion 4: {len(offenders)} row(s) missing citation_id: {offenders}"]


def run_fiche_assertions(fiches: Iterable[FicheRow], articles: Iterable[ArticleRow]) -> None:
    """Run assertion 5 — every fiche's `section_ids` resolve into the article corpus.

    `fiches` and `articles` are each consumed once — callers passing a generator should
    list() it first if they need it again afterwards.
    """
    violations = _assert_fiche_sections_resolve(list(fiches), list(articles))
    if violations:
        raise CorpusAssertionError(
            f"{len(violations)} ingest assertion violation(s):\n" + "\n".join(violations)
        )


def _assert_fiche_sections_resolve(fiches: list[FicheRow], articles: list[ArticleRow]) -> list[str]:
    """Assertion 5 — every fiche has >=1 `section_ids` entry matching an article `sectionParentId`.

    Per-fiche, not per-entry: a fiche's `<dc:source>` commonly names several `LEGISCTA`
    sections, and not all of them need carry an in-force article of their own — some point
    at a higher section in the hierarchy than any article's immediate parent. What actually
    breaks expansion (SPEC §7.4) is a fiche where *none* of its sections resolve, so that is
    the condition checked here — the same nonempty-intersection test the scope rule itself
    used to select the fiche in the first place (SPEC §1.2).
    """
    article_section_ids = {row["sectionParentId"] for row in articles if row.get("sectionParentId")}
    offenders = sorted(
        row["fiche_id"]
        for row in fiches
        if not any(section_id in article_section_ids for section_id in row["section_ids"])
    )
    if not offenders:
        return []
    return [
        f"assertion 5: {len(offenders)} fiche(s) with no section_ids resolving to an "
        f"article section_id: {offenders}"
    ]


def run_article_chunk_assertions(rows: Iterable[ArticleRow]) -> None:
    """Run assertions 6–9 (SPEC §4.5) over each row's `chunk_article` output.

    `rows` is consumed once — callers passing a generator should list() it first if they
    need it again afterwards. Chunking every row costs real time (the BGE-M3 tokenizer, not
    a character approximation, per assertion 7's own requirement), so this is a separate,
    heavier gate from `run_article_assertions` — run it where the chunker itself is what's
    under test, not on every corpus-shape check.
    """
    chunked = [(row, chunk_article(row)) for row in rows]
    violations: list[str] = [
        *_assert_every_article_has_a_chunk(chunked),
        *_assert_no_chunk_over_band(chunked),
        *_assert_non_stub_chunks_meet_floor(chunked),
        *_assert_no_content_loss(chunked),
    ]
    if violations:
        raise CorpusAssertionError(
            f"{len(violations)} ingest assertion violation(s):\n" + "\n".join(violations)
        )


def _assert_every_article_has_a_chunk(chunked: _Chunked) -> list[str]:
    """Assertion 6 — every article yields ≥ 1 chunk.

    Chunking always returns at least the stub path for a body that strips to nothing, so a
    zero-chunk row can only mean a chunker regression — this assertion exists to catch it
    at the corpus/chunker boundary rather than as a silently empty document downstream.
    """
    offenders = sorted(row["cid"] for row, chunks in chunked if not chunks)
    if not offenders:
        return []
    return [f"assertion 6: {len(offenders)} article(s) with zero chunks: {offenders}"]


def _assert_no_chunk_over_band(chunked: _Chunked) -> list[str]:
    """Assertion 7 — no chunk exceeds the 512-token band under the BGE-M3 tokenizer."""
    offenders = sorted({row["cid"] for row, chunks in chunked for chunk in chunks if chunk.tokens > BAND})
    if not offenders:
        return []
    return [f"assertion 7: {len(offenders)} article(s) with a chunk over {BAND} tokens: {offenders}"]


def _assert_non_stub_chunks_meet_floor(chunked: _Chunked) -> list[str]:
    """Assertion 8 — every non-stub chunk has ≥ 32 tokens of text."""
    offenders = sorted(
        {row["cid"] for row, chunks in chunked for chunk in chunks if not chunk.is_stub and chunk.tokens < STUB_FLOOR}
    )
    if not offenders:
        return []
    return [f"assertion 8: {len(offenders)} article(s) with a non-stub chunk under {STUB_FLOOR} tokens: {offenders}"]


def _assert_no_content_loss(chunked: _Chunked) -> list[str]:
    """Assertion 9 — `sum(chunk tokens)` per document is within tolerance of the source body
    token count minus stripped `<table>` content.

    The lower bound is exact (`>=`, zero slack): every real token of table-stripped source
    text is kept somewhere across a document's chunks, verified once on the committed corpus
    (measured delta was never negative — splitting and stub padding only ever add tokens,
    never drop them). The upper bound is generous rather than tight, because it exists to
    catch an HTML-shape regression duplicating content wholesale, not to police the exact
    overhead of `<s>`/`</s>` per extra chunk or a stub's citation suffix.
    """
    offenders = []
    for row, chunks in chunked:
        source_tokens = count_tokens(extract_text(row["texteHtml"]))
        summed_tokens = sum(chunk.tokens for chunk in chunks)
        split_overhead = max(0, len(chunks) - 1) * SPECIAL_TOKEN_COUNT
        stub_budget = _STUB_PADDING_BUDGET if any(chunk.is_stub for chunk in chunks) else 0
        upper_bound = source_tokens + split_overhead + stub_budget
        if not (source_tokens <= summed_tokens <= upper_bound):
            offenders.append(row["cid"])
    offenders.sort()
    if not offenders:
        return []
    return [
        f"assertion 9: {len(offenders)} article(s) where chunk tokens sum falls outside "
        f"tolerance of the source body token count: {offenders}"
    ]
