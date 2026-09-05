"""SPEC §9.1, §12.7, ADR-0015 — the rung-1 arm: path selection, merge, top-8, the fat object."""

import pytest
from conftest import CreateCollection, raw_point, stub_embed
from qdrant_client import QdrantClient

from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS
from rag.retrieval.pipeline import (
    ARTICLE_LEG,
    DEFAULT_RETRIEVAL_ARM,
    FICHE_LEG,
    RETRIEVAL_ARMS,
    rank_candidates,
    retrieve,
    retrieve_rung1,
)
from rag.retrieval.short_circuit import ShortCircuitPath


def _candidate(id_: str, score: float) -> Candidate:
    return Candidate(
        id=id_,
        score=score,
        register=Register.ARTICLE,
        payload={},
        provenance=frozenset({Provenance.SEARCH}),
    )


def test_rank_candidates_sorts_by_score_descending_and_caps_at_top_k() -> None:
    """Pure-Python, no Qdrant involved: this is the pipeline's own sort/cap logic, not a
    claim about the store's cosine ranking (SPEC §6.3 — `:memory:` is for plumbing only)."""
    candidates = [_candidate("a", 0.1), _candidate("b", 0.9), _candidate("c", 0.5)]

    ranked = rank_candidates(candidates, top_k=2)

    assert [c.id for c in ranked] == ["b", "c"]


def test_rank_candidates_defaults_to_top_k_eight() -> None:
    candidates = [_candidate(str(i), float(i)) for i in range(10)]

    assert len(rank_candidates(candidates)) == 8


def test_a_resolved_short_circuit_skips_search_and_has_no_candidate_pools(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [0.0, 0.0, 0.0, 1.0], {"lookup_key": "L113-2", "chunk_index": 0})],
    )

    result = retrieve_rung1(
        qdrant, stub_embed([1.0, 0.0, 0.0, 0.0]), "Que dit L113-2 ?", {"L113-2"}
    )

    assert result.short_circuit_path is ShortCircuitPath.RESOLVED
    assert result.candidate_pools == {}
    [context] = result.contexts
    assert context.provenance == frozenset({Provenance.LOOKUP})


def test_no_reference_falls_through_to_both_legs(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1"})])
    qdrant.upsert(
        ARTICLES_ALIAS, points=[raw_point(2, [0.9, 0.1, 0.0, 0.0], {"citation_id": "L113-2"})]
    )

    result = retrieve_rung1(
        qdrant,
        stub_embed([1.0, 0.0, 0.0, 0.0]),
        "Quelle franchise pour un dégât des eaux ?",
        set(),
    )

    assert result.short_circuit_path is ShortCircuitPath.NO_REFERENCE
    assert set(result.candidate_pools) == {FICHE_LEG, ARTICLE_LEG}
    assert len(result.candidate_pools[FICHE_LEG]) == 1
    assert len(result.candidate_pools[ARTICLE_LEG]) == 1
    # Membership only — which of the two ranks first is a ranking claim about the store's
    # own cosine math, reserved for the `qdrant_server` fixture (SPEC §6.3).
    assert {c.register for c in result.contexts} == {Register.FICHE, Register.ARTICLE}
    assert all(c.provenance == frozenset({Provenance.SEARCH}) for c in result.contexts)


def test_membership_failure_falls_through_to_search_exactly_like_no_reference(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2"})]
    )

    result = retrieve_rung1(
        qdrant, stub_embed([1.0, 0.0, 0.0, 0.0]), "Que dit L999-9 ?", {"L113-2"}
    )

    assert result.short_circuit_path is ShortCircuitPath.MEMBERSHIP_FAILED
    assert set(result.candidate_pools) == {FICHE_LEG, ARTICLE_LEG}


def test_final_contexts_are_capped_at_top_k(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(i, [1.0, 0.0, 0.0, 0.0], {"citation_id": f"L{i}"}) for i in range(10)],
    )

    result = retrieve_rung1(qdrant, stub_embed([1.0, 0.0, 0.0, 0.0]), "une question ouverte", set())

    assert len(result.contexts) == 8


def test_retrieve_dispatches_to_the_named_arm(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)

    result = retrieve(
        qdrant, stub_embed([1.0, 0.0, 0.0, 0.0]), "une question ouverte", set(), arm="rung1"
    )

    assert result.short_circuit_path is ShortCircuitPath.NO_REFERENCE


def test_retrieve_rejects_an_unknown_arm(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    with pytest.raises(KeyError):
        retrieve(qdrant, stub_embed([1.0, 0.0, 0.0, 0.0]), "q", set(), arm="rung99")


def test_rung1_is_the_default_arm_and_is_registered() -> None:
    assert DEFAULT_RETRIEVAL_ARM == "rung1"
    assert RETRIEVAL_ARMS["rung1"] is retrieve_rung1
