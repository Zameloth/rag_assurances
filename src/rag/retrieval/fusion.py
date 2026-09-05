"""Per-leg weighted hybrid fusion, client-side (SPEC §9.3, ADR-0006, #29).

Dense and sparse are two independent `query_points` calls per leg (`rag.retrieval.legs`);
this module is what turns those two ranked pools into the one pool a leg actually returns.
Qdrant's own RRF fuses by rank position with no weight knob, which is exactly what SPEC
§9.3 rules out — the MIRACL/MLDR evidence is a *per-leg weight*, and RRF has nowhere to put
one. `fuse_scores` is therefore a plain weighted sum of the two raw scores (the same shape
BGE-M3's own reference `compute_score` uses for its dense/sparse/colbert blend), not a
rank-based method — cosine and sparse-dot-product are already comparable magnitudes here
because both come from the same BGE-M3 forward pass this project standardised on.

**Weights are a named constant plus an injectable parameter** (`FICHE_LEG_WEIGHTS` /
`ARTICLE_LEG_WEIGHTS`, passed into `hybrid_leg` rather than read from inside it) — the same
shape ADR-0015 already chose for which retrieval arm runs, and for the same reason:
CONTEXT.md's index-bearing/runtime split calls per-leg weights "runtime... app config", and
SPEC §16.3's `.env` table is the exhaustive list of what qualifies as an environment fact,
which a tuning knob is not.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from qdrant_client import QdrantClient, models

from rag.retrieval.candidates import Candidate, Register
from rag.retrieval.legs import search_leg, search_leg_sparse

__all__ = [
    "ARTICLE_LEG_WEIGHTS",
    "FICHE_LEG_WEIGHTS",
    "LegWeights",
    "fuse_scores",
    "hybrid_leg",
]


@dataclass(frozen=True)
class LegWeights:
    """How much a leg's fused score owes to each vector kind. Not required to sum to 1 —
    `fuse_scores` only ever adds weighted terms, never normalises against their sum."""

    dense: float
    sparse: float


# SPEC §9.3's regime table (French column): MIRACL-fr (short passages) favours dense,
# MLDR-fr (long documents) favours sparse, and the two invert. 2:1 mirrors BGE-M3's own
# reference `compute_score` default weighting of dense against sparse (colbert dropped,
# proportionally renormalised) rather than an arbitrary round number — a starting point
# for the ladder to tune, not a measured-for-this-corpus value.
FICHE_LEG_WEIGHTS = LegWeights(dense=2 / 3, sparse=1 / 3)
ARTICLE_LEG_WEIGHTS = LegWeights(dense=1 / 3, sparse=2 / 3)


def hybrid_leg(
    client: QdrantClient,
    register: Register,
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    weights: LegWeights,
    *,
    limit: int,
) -> list[Candidate]:
    """One leg, hybrid: dense and sparse `query_points` at `limit` each, fused client-side
    by `weights` and capped back down to `limit` (SPEC §9.2 fixes both legs at top-20
    regardless of rung, so the fused pool stays that same depth, not the union's)."""
    dense_pool = search_leg(client, register, dense_vector, limit)
    sparse_pool = search_leg_sparse(client, register, sparse_vector, limit)
    return fuse_scores(dense_pool, sparse_pool, weights, limit=limit)


def fuse_scores(
    dense_pool: list[Candidate], sparse_pool: list[Candidate], weights: LegWeights, *, limit: int
) -> list[Candidate]:
    """Dedupe `dense_pool`/`sparse_pool` by point id, scoring each id as the weighted sum
    of whichever raw score(s) it was found with — zero contribution from a side that didn't
    return the id at all, not a penalty and not an imputed value. Provenance unions the same
    way `merge_candidates` does (SPEC §7.5); payload/register come from whichever pool is
    seen first, since a single point id carries one payload regardless of which query found
    it. Sorted by fused score descending, capped at `limit`.
    """
    fused: dict[str, Candidate] = {}
    for candidate in dense_pool:
        fused[candidate.id] = replace(candidate, score=weights.dense * candidate.score)
    for candidate in sparse_pool:
        contribution = weights.sparse * candidate.score
        existing = fused.get(candidate.id)
        if existing is None:
            fused[candidate.id] = replace(candidate, score=contribution)
        else:
            fused[candidate.id] = replace(
                existing,
                score=existing.score + contribution,
                provenance=existing.provenance | candidate.provenance,
            )
    ranked = sorted(fused.values(), key=lambda candidate: candidate.score, reverse=True)
    return ranked[:limit]
