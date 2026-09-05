"""One search leg's two raw `qdrant-client` queries against a stable alias (SPEC §6.1, §9.2,
§9.3, #28, #29).

`search_leg` (dense) and `search_leg_sparse` (BGE-M3's learned lexical weights) each issue
one `query_points` call per collection and return plain `Candidate` objects — combining the
two into one leg's fused pool is `rag.retrieval.fusion.hybrid_leg`'s job, not this module's.
Rung 1 (#28) calls `search_leg` alone; rung 2 (#29) calls both. Neither function is a
LangChain component — SPEC §6.1 reserves that wrapping for the `BaseRetriever` subclass
itself.

**Aliases only.** `ARTICLES_ALIAS` / `FICHES_ALIAS` (re-exported from `rag.ingest.arms`,
their single defining home — SPEC §6.4) are never a physical `__m3__c512__v1`-style arm
name. Switching which arm serves traffic is an alias flip (`rag.ingest.arms.flip_alias`);
nothing here would need to change for that to take effect.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from rag.ingest.arms import ARTICLES_ALIAS, FICHES_ALIAS
from rag.retrieval.candidates import Candidate, Provenance, Register

__all__ = ["ARTICLES_ALIAS", "FICHES_ALIAS", "search_leg", "search_leg_sparse"]

_REGISTER_ALIAS = {
    Register.FICHE: FICHES_ALIAS,
    Register.ARTICLE: ARTICLES_ALIAS,
}


def search_leg(
    client: QdrantClient, register: Register, dense_vector: list[float], limit: int
) -> list[Candidate]:
    """Dense-only `query_points` against `register`'s alias, `limit` hits, scored by cosine
    similarity (SPEC §6.3's exact search — `indexing_threshold=0` on every arm makes this
    exhaustive regardless of collection size).

    Every hit is annotated `provenance={Provenance.SEARCH}` here, on the way out — SPEC
    §7.5 attaches `register`/`provenance` at the retriever boundary, never earlier and
    never by storing them.
    """
    return _search(client, register, dense_vector, using="dense", limit=limit)


def search_leg_sparse(
    client: QdrantClient, register: Register, sparse_vector: models.SparseVector, limit: int
) -> list[Candidate]:
    """`search_leg`'s sibling for BGE-M3's learned sparse half (SPEC §9.3, #29): the same
    alias, the same `limit` hits, the same annotation contract — scored by the store's
    sparse dot product instead of dense cosine.
    """
    return _search(client, register, sparse_vector, using="sparse", limit=limit)


def _search(
    client: QdrantClient,
    register: Register,
    query: list[float] | models.SparseVector,
    *,
    using: str,
    limit: int,
) -> list[Candidate]:
    alias = _REGISTER_ALIAS[register]
    hits = client.query_points(
        collection_name=alias,
        query=query,
        using=using,
        limit=limit,
        with_payload=True,
    ).points
    return [
        Candidate(
            id=str(hit.id),
            score=hit.score,
            register=register,
            payload=hit.payload or {},
            provenance=frozenset({Provenance.SEARCH}),
        )
        for hit in hits
    ]
