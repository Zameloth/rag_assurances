"""The short-circuit's metadata-lookup path: `lookup_key` in, article chunks out (SPEC §9.1, §7.2, #28).

Two jobs, both native `qdrant-client` reads against the `articles` alias — never search,
never a physical collection name:

- `load_lookup_keys` — the membership set `rag.retrieval.short_circuit.resolve_short_circuit`
  checks a scanned reference against. Loaded fresh per process rather than cached across
  the retriever's lifetime; a corpus refresh (SPEC §3.3) is a manual, reviewed commit, not
  something the running process needs to detect on its own.
- `lookup_article_chunks_by_key` — resolves a `RESOLVED` short-circuit into the article's
  chunks, ordered the way they were written (`chunk_index`), skipping the search legs
  entirely (SPEC §9.1's diagram: "Metadata lookup on articles (no search, no
  condensation)").

Both filter on `lookup_key`, which `rag.ingest.arms.ensure_articles_collection` indexes
(SPEC §7.2) — a keyword index on `articles`, none on `fiches`, matching the fact that only
`articles` carries the field at all. Neither takes an `alias` override: SPEC §9.1's short-
circuit is defined against `articles` specifically, and a parameter that could point this
at a physical collection name would be the one crack in "the retriever always reads
through the aliases."
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.legs import ARTICLES_ALIAS

__all__ = ["lookup_article_chunks_by_key", "load_lookup_keys"]

# Payload-only, no vectors — this is a metadata scan, not a similarity search, and pulling
# vectors back would cost bandwidth for values nothing here reads.
_SCROLL_PAGE_SIZE = 256


def load_lookup_keys(client: QdrantClient) -> frozenset[str]:
    """The full `lookup_key` membership set (SPEC §9.1), read from `articles`.

    Scans every point rather than filtering server-side for non-null values: the corpus is
    small (2,801 article points, SPEC §4.4) and a plain scroll keeps this function honest
    about what it does — read every `lookup_key`, keep the ones that aren't `None` (the 21
    prose annexe labels, SPEC §7.3) — without depending on a null-filter shape that isn't
    exercised anywhere else in the codebase.
    """
    keys: set[str] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=ARTICLES_ALIAS,
            with_payload=["lookup_key"],
            with_vectors=False,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
        )
        for record in records:
            key = (record.payload or {}).get("lookup_key")
            if key is not None:
                keys.add(key)
        if offset is None:
            break
    return frozenset(keys)


def lookup_article_chunks_by_key(client: QdrantClient, lookup_key: str) -> list[Candidate]:
    """Every chunk of the article whose `lookup_key` matches, ordered by `chunk_index`.

    An article's chunks all carry the same `lookup_key` (`build_article_payload` computes
    it once per row and stamps every chunk with it), so a single-value filter can return
    more than one point — the short-circuit resolves to the whole article, not just its
    first chunk. Paginates the same way `load_lookup_keys` does: nothing measured bounds
    chunks-per-article below what one scroll page could miss, so a second page must be
    read rather than assumed away.

    Scored `1.0` and tagged `provenance={Provenance.LOOKUP}`: a metadata lookup has no
    similarity score to report, and `1.0` reads as "certain" rather than as a comparable
    cosine value — this path is never merged with a search pool (SPEC §9.1: short-circuit
    skips search entirely), so no comparison is ever made against it.
    """
    scroll_filter = models.Filter(
        must=[models.FieldCondition(key="lookup_key", match=models.MatchValue(value=lookup_key))]
    )
    candidates: list[Candidate] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=ARTICLES_ALIAS,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
        )
        candidates.extend(
            Candidate(
                id=str(record.id),
                score=1.0,
                register=Register.ARTICLE,
                payload=record.payload or {},
                provenance=frozenset({Provenance.LOOKUP}),
            )
            for record in records
        )
        if offset is None:
            break
    candidates.sort(key=lambda candidate: candidate.payload.get("chunk_index", 0))
    return candidates
