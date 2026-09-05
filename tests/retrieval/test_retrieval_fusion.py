"""SPEC §9.3, ADR-0006, #29 — per-leg weighted hybrid fusion, client-side."""

from conftest import CreateCollection, raw_point
from qdrant_client import QdrantClient, models

from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.fusion import (
    ARTICLE_LEG_WEIGHTS,
    FICHE_LEG_WEIGHTS,
    LegWeights,
    fuse_scores,
    hybrid_leg,
)
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS


def _candidate(id_: str, score: float, register: Register = Register.ARTICLE) -> Candidate:
    return Candidate(
        id=id_,
        score=score,
        register=register,
        payload={},
        provenance=frozenset({Provenance.SEARCH}),
    )


def test_fuse_scores_weights_a_hit_present_in_both_pools() -> None:
    dense_pool = [_candidate("a", 1.0)]
    sparse_pool = [_candidate("a", 0.5)]

    [fused] = fuse_scores(dense_pool, sparse_pool, LegWeights(dense=0.7, sparse=0.3), limit=20)

    assert fused.id == "a"
    assert fused.score == 0.7 * 1.0 + 0.3 * 0.5


def test_fuse_scores_weights_a_hit_present_in_only_one_pool() -> None:
    dense_only = [_candidate("a", 1.0)]

    [fused] = fuse_scores(dense_only, [], LegWeights(dense=0.7, sparse=0.3), limit=20)

    # No sparse contribution to average against — a dense-only hit is not penalised for
    # missing the other leg, it just carries its own weighted share.
    assert fused.score == 0.7 * 1.0


def test_fuse_scores_weights_a_hit_present_in_the_sparse_pool_only() -> None:
    sparse_only = [_candidate("a", 1.0)]

    [fused] = fuse_scores([], sparse_only, LegWeights(dense=0.7, sparse=0.3), limit=20)

    assert fused.score == 0.3 * 1.0


def test_fuse_scores_unions_provenance_for_a_hit_present_in_both_pools() -> None:
    dense = Candidate(
        id="a", score=1.0, register=Register.ARTICLE, payload={},
        provenance=frozenset({Provenance.SEARCH}),
    )
    sparse = Candidate(
        id="a", score=1.0, register=Register.ARTICLE, payload={},
        provenance=frozenset({Provenance.EXPANSION}),
    )

    [fused] = fuse_scores([dense], [sparse], LegWeights(dense=0.5, sparse=0.5), limit=20)

    assert fused.provenance == frozenset({Provenance.SEARCH, Provenance.EXPANSION})


def test_fuse_scores_sorts_by_fused_score_descending_and_caps_at_limit() -> None:
    dense_pool = [_candidate(str(i), float(i)) for i in range(10)]

    fused = fuse_scores(dense_pool, [], LegWeights(dense=1.0, sparse=0.0), limit=3)

    assert [c.id for c in fused] == ["9", "8", "7"]


def test_fiche_leg_weights_are_dense_leaning() -> None:
    assert FICHE_LEG_WEIGHTS.dense > FICHE_LEG_WEIGHTS.sparse


def test_article_leg_weights_are_sparse_leaning() -> None:
    assert ARTICLE_LEG_WEIGHTS.sparse > ARTICLE_LEG_WEIGHTS.dense


def test_hybrid_leg_queries_both_vector_kinds_and_fuses_the_results(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[
            raw_point(1, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2"}),
            raw_point(2, [0.0, 1.0, 0.0, 0.0], {"citation_id": "L113-3"}),
        ],
    )

    pool = hybrid_leg(
        qdrant,
        Register.ARTICLE,
        dense_vector=[1.0, 0.0, 0.0, 0.0],
        sparse_vector=models.SparseVector(indices=[1], values=[0.5]),
        weights=LegWeights(dense=0.7, sparse=0.3),
        limit=20,
    )

    # Membership only — SPEC §6.3 reserves ranking-number claims for the `qdrant_server`
    # fixture, never `:memory:` (see `tests/conftest.py`'s `qdrant` fixture docstring).
    assert {c.payload["citation_id"] for c in pool} == {"L113-2", "L113-3"}
    assert all(c.register is Register.ARTICLE for c in pool)


def test_hybrid_leg_respects_the_limit(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    qdrant.upsert(
        FICHES_ALIAS,
        points=[raw_point(i, [1.0, 0.0, 0.0, 0.0], {}) for i in range(5)],
    )

    pool = hybrid_leg(
        qdrant,
        Register.FICHE,
        dense_vector=[1.0, 0.0, 0.0, 0.0],
        sparse_vector=models.SparseVector(indices=[1], values=[0.5]),
        weights=FICHE_LEG_WEIGHTS,
        limit=2,
    )

    assert len(pool) == 2
