"""Ingest assertions 1–4 (SPEC §7.4).

These gate the article corpus, not only a later ingest run: the fetch script (a dev tool,
SPEC §3.3) calls `run_article_assertions` before it writes `articles.jsonl`, and the test
suite calls it again against the committed file so a corpus that fails its own assertions
can never land on `main`. Assertion 5 (fiche `section_ids` resolve to an article
`section_id`) needs the fiches corpus and is out of scope here.

Every assertion raises `CorpusAssertionError` with every violation it found, not just the
first — a refresh that fails for three unrelated reasons should say so once, not three
runs apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rag.ingest.lookup_key import normalize_lookup_key

__all__ = ["CorpusAssertionError", "run_article_assertions"]

ArticleRow = Mapping[str, Any]


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
