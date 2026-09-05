"""`<dc:source>` expansion — the rung-3 arm's spine (SPEC §9.2, ADR-0006, #30).

The top fiches from the fiche leg carry `section_ids` (read here, never filtered on — SPEC
§7.1, CONTEXT.md's `section_ids` entry). Their union is the candidate set this module hands
to a **filtered vector search** over `articles`: `MatchAny` on the indexed `section_id`
gates which points are eligible, and the query's own dense vector orders them — a
`scroll` would order by point id (UUIDv5, uncorrelated with relevance) and make rung 3
partly measure the hash function (SPEC §9.2), which is exactly what a filtered
`query_points` call avoids.

This does not reinstate the weak consumer→legal hop (SPEC §9.2): similarity is demoted from
**gate** (deciding which articles are even considered) to **sort order** over a set DILA has
already editorially certified as the fiche's legal basis.
"""

from __future__ import annotations

from collections.abc import Iterable

from qdrant_client import QdrantClient, models

from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.legs import ARTICLES_ALIAS

__all__ = [
    "EXPANSION_CAP",
    "EXPANSION_FICHE_DEPTH",
    "expand",
    "expand_by_section",
    "section_ids_from_fiches",
    "top_fiches",
]

# SPEC §9.2: "the top 3 fiches expand into the sections they cite" — the fiche depth this
# rung's expansion reads from, not a physical count of anything stored.
EXPANSION_FICHE_DEPTH = 3

# SPEC §9.2: "capped at 40" — measured against the corpus (median 3 articles/section, p90
# 10, max 43) to bind rarely rather than to fit a target.
EXPANSION_CAP = 40


def top_fiches(fiche_pool: list[Candidate], depth: int) -> list[Candidate]:
    """The first `depth` fiches of an already-ranked pool — `fiche_pool` is `hybrid_leg`'s
    (or `search_leg`'s) own output, sorted by score descending, so "top" is positional, not
    a second sort this function performs."""
    return fiche_pool[:depth]


def section_ids_from_fiches(fiches: Iterable[Candidate]) -> list[str]:
    """The union of `section_ids` across `fiches`, read off each fiche chunk's payload and
    never filtered on (SPEC §7.1) — deduped, order-preserving so the resulting `MatchAny`
    list is deterministic for a given input rather than depending on set iteration order."""
    section_ids: list[str] = []
    seen: set[str] = set()
    for fiche in fiches:
        for section_id in fiche.payload.get("section_ids") or []:
            if section_id not in seen:
                seen.add(section_id)
                section_ids.append(section_id)
    return section_ids


def expand_by_section(
    client: QdrantClient,
    dense_vector: list[float],
    section_ids: list[str],
    *,
    cap: int = EXPANSION_CAP,
) -> list[Candidate]:
    """The filtered vector search itself (SPEC §9.2): `MatchAny` on `articles`' indexed
    `section_id` gates the candidate set, the query's dense vector orders it, `cap` bounds
    it. Empty `section_ids` returns no candidates rather than issuing a filterless query
    that would silently degrade into an unfiltered top-`cap` search.

    Every hit is annotated `provenance={Provenance.EXPANSION}` — SPEC §7.5 attaches
    `register`/`provenance` at the retriever boundary, never earlier and never by storing
    them, the same contract `rag.retrieval.legs._search` follows.
    """
    if not section_ids:
        return []
    query_filter = models.Filter(
        must=[models.FieldCondition(key="section_id", match=models.MatchAny(any=section_ids))]
    )
    hits = client.query_points(
        collection_name=ARTICLES_ALIAS,
        query=dense_vector,
        using="dense",
        query_filter=query_filter,
        limit=cap,
        with_payload=True,
    ).points
    return [
        Candidate(
            id=str(hit.id),
            score=hit.score,
            register=Register.ARTICLE,
            payload=hit.payload or {},
            provenance=frozenset({Provenance.EXPANSION}),
        )
        for hit in hits
    ]


def expand(
    client: QdrantClient,
    fiche_pool: list[Candidate],
    dense_vector: list[float],
    *,
    depth: int = EXPANSION_FICHE_DEPTH,
    cap: int = EXPANSION_CAP,
) -> list[Candidate]:
    """The rung-3 arm's whole expansion step: top `depth` fiches -> their sections -> a
    filtered vector search within them, capped at `cap`. `depth`/`cap` are real parameters,
    not just internal ones this function happens to take — CONTEXT.md names both "runtime...
    app config", and this ticket's own acceptance criterion is that the arm stays ablatable
    through them, the same shape `hybrid_leg`'s `weights` already takes (#29, ADR-0016)."""
    fiches = top_fiches(fiche_pool, depth)
    section_ids = section_ids_from_fiches(fiches)
    return expand_by_section(client, dense_vector, section_ids, cap=cap)
