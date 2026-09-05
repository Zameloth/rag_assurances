"""One search leg: a raw `qdrant-client` dense query against a stable alias (SPEC §6.1, §9.2, #28).

Rung 1 is dense-only (SPEC §12.7's "naive baseline" row) — no sparse leg, no per-leg
weighting, no client-side fusion math. Those land with the hybrid-legs ticket (#29); this
module is the seam it plugs into, not a pre-implementation of it. `search_leg` issues one
`query_points` call per collection and returns plain `Candidate` objects, so nothing here
is a LangChain component — SPEC §6.1 reserves that wrapping for the `BaseRetriever`
subclass itself.

**Aliases only.** `ARTICLES_ALIAS` / `FICHES_ALIAS` (re-exported from `rag.ingest.arms`,
their single defining home — SPEC §6.4) are never a physical `__m3__c512__v1`-style arm
name. Switching which arm serves traffic is an alias flip (`rag.ingest.arms.flip_alias`);
nothing here would need to change for that to take effect.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from rag.ingest.arms import ARTICLES_ALIAS, FICHES_ALIAS
from rag.retrieval.candidates import Candidate, Provenance, Register

__all__ = ["ARTICLES_ALIAS", "FICHES_ALIAS", "search_leg"]

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
    alias = _REGISTER_ALIAS[register]
    hits = client.query_points(
        collection_name=alias,
        query=dense_vector,
        using="dense",
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
