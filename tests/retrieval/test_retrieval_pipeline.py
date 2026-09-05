"""SPEC §9.1, §12.7, ADR-0015 — the rung-1 arm: path selection, merge, top-8, the fat object."""

import pytest
from conftest import CreateCollection, raw_point, stub_embed, stub_embed_hybrid
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.fusion import LegWeights
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS
from rag.retrieval.pipeline import (
    ARTICLE_LEG,
    DEFAULT_RETRIEVAL_ARM,
    EXPANSION_POOL,
    FICHE_LEG,
    RETRIEVAL_ARMS,
    rank_candidates,
    retrieve,
    retrieve_rung1,
    retrieve_rung2,
    retrieve_rung3,
    retrieve_rung4,
    retrieve_rung5,
)
from rag.retrieval.rerank import RerankFn
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


def test_rung2_short_circuit_behaves_exactly_like_rung1(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [0.0, 0.0, 0.0, 1.0], {"lookup_key": "L113-2", "chunk_index": 0})],
    )

    result = retrieve_rung2(
        qdrant, stub_embed([1.0, 0.0, 0.0, 0.0]), "Que dit L113-2 ?", {"L113-2"}
    )

    assert result.short_circuit_path is ShortCircuitPath.RESOLVED
    assert result.candidate_pools == {}
    [context] = result.contexts
    assert context.provenance == frozenset({Provenance.LOOKUP})


def test_rung2_hybridizes_both_legs_and_returns_hybrid_leg_pools(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1"})])
    qdrant.upsert(
        ARTICLES_ALIAS, points=[raw_point(2, [0.9, 0.1, 0.0, 0.0], {"citation_id": "L113-2"})]
    )

    result = retrieve_rung2(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "Quelle franchise pour un dégât des eaux ?",
        set(),
    )

    assert result.short_circuit_path is ShortCircuitPath.NO_REFERENCE
    assert set(result.candidate_pools) == {FICHE_LEG, ARTICLE_LEG}
    assert len(result.candidate_pools[FICHE_LEG]) == 1
    assert len(result.candidate_pools[ARTICLE_LEG]) == 1
    assert {c.register for c in result.contexts} == {Register.FICHE, Register.ARTICLE}
    assert all(c.provenance == frozenset({Provenance.SEARCH}) for c in result.contexts)


def test_rung2_final_contexts_are_capped_at_top_k(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(i, [1.0, 0.0, 0.0, 0.0], {"citation_id": f"L{i}"}) for i in range(10)],
    )

    result = retrieve_rung2(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
    )

    assert len(result.contexts) == 8


def test_rung2_is_registered_but_rung1_stays_the_default_arm() -> None:
    assert RETRIEVAL_ARMS["rung2"] is retrieve_rung2
    assert DEFAULT_RETRIEVAL_ARM == "rung1"
    assert RETRIEVAL_ARMS[DEFAULT_RETRIEVAL_ARM] is retrieve_rung1


def test_rung2_leg_weights_are_overridable_by_the_caller(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """`fiche_weights`/`article_weights` are real parameters, not just internal ones
    `hybrid_leg` happens to take — a caller can override the defaults without editing
    `fusion.py`'s constants (ADR-0016)."""
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        FICHES_ALIAS,
        points=[
            raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1"}),
            raw_point(2, [0.0, 1.0, 0.0, 0.0], {"fiche_id": "F2"}),
        ],
    )

    all_sparse = LegWeights(dense=0.0, sparse=1.0)
    result = retrieve_rung2(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
        fiche_weights=all_sparse,
    )

    # Dense contributes nothing under an all-sparse override, so every hit fuses to the
    # same sparse-only score (the sparse dot product of the identical query/point vectors,
    # 0.5*0.5) regardless of dense similarity — membership, not ranking (SPEC §6.3), but the
    # flat score is exactly what a zero dense weight predicts.
    fiche_scores = {c.score for c in result.candidate_pools[FICHE_LEG]}
    assert fiche_scores == {0.25}


def test_rung3_short_circuit_behaves_exactly_like_rung1(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [0.0, 0.0, 0.0, 1.0], {"lookup_key": "L113-2", "chunk_index": 0})],
    )

    result = retrieve_rung3(
        qdrant, stub_embed([1.0, 0.0, 0.0, 0.0]), "Que dit L113-2 ?", {"L113-2"}
    )

    assert result.short_circuit_path is ShortCircuitPath.RESOLVED
    assert result.candidate_pools == {}
    [context] = result.contexts
    assert context.provenance == frozenset({Provenance.LOOKUP})


def test_rung3_adds_an_expansion_pool_alongside_the_two_legs(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        FICHES_ALIAS,
        points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1", "section_ids": ["S1"]})],
    )
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(2, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2", "section_id": "S1"})],
    )

    result = retrieve_rung3(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
    )

    assert set(result.candidate_pools) == {FICHE_LEG, ARTICLE_LEG, EXPANSION_POOL}
    assert len(result.candidate_pools[EXPANSION_POOL]) == 1
    assert result.candidate_pools[EXPANSION_POOL][0].provenance == frozenset({Provenance.EXPANSION})


def test_rung3_an_article_reached_by_both_the_article_leg_and_expansion_carries_both_provenances(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """SPEC §7.5 / this ticket's own acceptance criterion — the union survives the merge
    regardless of which pool `merge_candidates` sees first."""
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        FICHES_ALIAS,
        points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1", "section_ids": ["S1"]})],
    )
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(2, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2", "section_id": "S1"})],
    )

    result = retrieve_rung3(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
    )

    # Present in both `article_pool` (unfiltered hybrid search over all of `articles`) and
    # `expansion_pool` (the same point, filtered on its own `section_id`) — the merged
    # context must carry both, not whichever pool happened to be merged first.
    [article_context] = [c for c in result.contexts if c.register is Register.ARTICLE]
    assert article_context.provenance == frozenset({Provenance.SEARCH, Provenance.EXPANSION})


def test_rung3_expansion_depth_zero_disables_expansion(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """`expansion_depth`/`expansion_cap` are real parameters, the same shape
    `fiche_weights`/`article_weights` already take (#29) — this ticket's own acceptance
    criterion is that the arm stays ablatable through them."""
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        FICHES_ALIAS,
        points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1", "section_ids": ["S1"]})],
    )
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(2, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2", "section_id": "S1"})],
    )

    result = retrieve_rung3(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
        expansion_depth=0,
    )

    assert result.candidate_pools[EXPANSION_POOL] == []


def test_rung3_final_contexts_are_capped_at_top_k(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(i, [1.0, 0.0, 0.0, 0.0], {"citation_id": f"L{i}"}) for i in range(10)],
    )

    result = retrieve_rung3(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
    )

    assert len(result.contexts) == 8


def test_rung3_is_registered_but_rung1_stays_the_default_arm() -> None:
    assert RETRIEVAL_ARMS["rung3"] is retrieve_rung3
    assert DEFAULT_RETRIEVAL_ARM == "rung1"
    assert RETRIEVAL_ARMS[DEFAULT_RETRIEVAL_ARM] is retrieve_rung1


def _reversing_rerank(query: str, candidates: list[Candidate]) -> list[Candidate]:
    """A fake `RerankFn`: reverses the merged pool and stamps a distinctive score on each
    candidate — so a test can tell "rung 4 reordered/rescored this" from "rung 3's own
    order/score survived untouched"."""
    reversed_pool = list(reversed(candidates))
    return [
        Candidate(
            id=c.id,
            score=100.0 + i,
            register=c.register,
            payload=c.payload,
            provenance=c.provenance,
        )
        for i, c in enumerate(reversed_pool)
    ]


def test_rung4_short_circuit_behaves_exactly_like_rung1(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [0.0, 0.0, 0.0, 1.0], {"lookup_key": "L113-2", "chunk_index": 0})],
    )

    result = retrieve_rung4(
        qdrant, stub_embed([1.0, 0.0, 0.0, 0.0]), "Que dit L113-2 ?", {"L113-2"}
    )

    assert result.short_circuit_path is ShortCircuitPath.RESOLVED
    assert result.candidate_pools == {}
    [context] = result.contexts
    assert context.provenance == frozenset({Provenance.LOOKUP})


def test_rung4_reranks_the_merged_pool_and_keeps_rung3_s_candidate_pool_shape(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        FICHES_ALIAS,
        points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1", "section_ids": ["S1"]})],
    )
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(2, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2", "section_id": "S1"})],
    )

    result = retrieve_rung4(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
        rerank_fn=_reversing_rerank,
    )

    # candidate_pools is rung 3's own shape, unaffected by reranking (SPEC §9.4: reranking
    # is one step over the already-merged pool, not a fifth candidate pool).
    assert set(result.candidate_pools) == {FICHE_LEG, ARTICLE_LEG, EXPANSION_POOL}
    # Every returned context carries the fake reranker's stamped score, proving rung 4's
    # rerank step (not rung 3's fused score) is what `rank_candidates` sorted.
    assert all(c.score >= 100.0 for c in result.contexts)


def test_rung4_final_contexts_are_capped_at_top_k(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(i, [1.0, 0.0, 0.0, 0.0], {"citation_id": f"L{i}"}) for i in range(10)],
    )

    result = retrieve_rung4(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
        rerank_fn=_reversing_rerank,
    )

    assert len(result.contexts) == 8


def test_rung4_rerank_fn_defaults_to_none_and_resolves_the_real_reranker(
    monkeypatch: pytest.MonkeyPatch, qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """Proves the production seam: with no `rerank_fn` injected, `retrieve_rung4` reaches
    `rag.retrieval.rerank.rerank` itself, bound to `reranker_model`/`reranker_backend` — the
    same shape `expansion_depth`/`fiche_weights` already take (#29, #30), and this ticket's
    own acceptance criterion that the cheap arm/int8 ONNX are reachable without a code
    change."""
    import rag.retrieval.pipeline as pipeline_module
    from rag.retrieval.rerank import RerankerBackend

    seen: dict[str, object] = {}

    def _fake_rerank(
        query: str, candidates: list[Candidate], *, model_id: str, backend: RerankerBackend
    ) -> list[Candidate]:
        seen["query"] = query
        seen["model_id"] = model_id
        seen["backend"] = backend
        return candidates

    monkeypatch.setattr(pipeline_module, "rerank", _fake_rerank)
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)

    retrieve_rung4(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
        reranker_model="Alibaba-NLP/gte-multilingual-reranker-base",
        reranker_backend=RerankerBackend.ONNX_INT8,
    )

    assert seen["query"] == "une question ouverte"
    assert seen["model_id"] == "Alibaba-NLP/gte-multilingual-reranker-base"
    assert seen["backend"] is RerankerBackend.ONNX_INT8


def test_rung4_is_registered_but_rung1_stays_the_default_arm() -> None:
    assert RETRIEVAL_ARMS["rung4"] is retrieve_rung4
    assert DEFAULT_RETRIEVAL_ARM == "rung1"
    assert RETRIEVAL_ARMS[DEFAULT_RETRIEVAL_ARM] is retrieve_rung1


def _fixed_score_rerank(scores: dict[str, float]) -> RerankFn:
    """A fake `RerankFn` that stamps a caller-chosen score onto each candidate by id,
    leaving provenance/register/payload untouched — so a rung-5 test can control exactly
    which candidates clear the relevance floor and in what order, independent of the fake
    store's own cosine numbers (SPEC §6.3)."""

    def _fn(query: str, candidates: list[Candidate]) -> list[Candidate]:
        return [
            Candidate(
                id=c.id,
                score=scores.get(c.id, c.score),
                register=c.register,
                payload=c.payload,
                provenance=c.provenance,
            )
            for c in candidates
        ]

    return _fn


def test_rung5_short_circuit_behaves_exactly_like_rung1(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [0.0, 0.0, 0.0, 1.0], {"lookup_key": "L113-2", "chunk_index": 0})],
    )

    result = retrieve_rung5(
        qdrant, stub_embed([1.0, 0.0, 0.0, 0.0]), "Que dit L113-2 ?", {"L113-2"}
    )

    assert result.short_circuit_path is ShortCircuitPath.RESOLVED
    assert result.candidate_pools == {}
    assert result.floor_met is None
    [context] = result.contexts
    assert context.provenance == frozenset({Provenance.LOOKUP})


def test_rung5_fills_fiche_and_article_slots_separately_not_a_shared_top_8(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """The failure SPEC §9.5 designs out: a naive shared top-8 cap over a pool this lopsided
    would return eight fiches and zero articles. Quota assembly must not."""
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        FICHES_ALIAS,
        points=[raw_point(i, [1.0, 0.0, 0.0, 0.0], {"fiche_id": f"F{i}"}) for i in range(6)],
    )
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[
            raw_point(100 + i, [1.0, 0.0, 0.0, 0.0], {"citation_id": f"L{i}"}) for i in range(2)
        ],
    )
    # Every fiche outscores every article post-rerank — naive top-8 would take all 6
    # fiches and both articles ranked below them, never leaving room to notice a shortfall,
    # but there are only 2 articles here regardless: the point is that quota reserves 4
    # article slots rather than letting fiches crowd every remaining slot.
    scores = {str(i): 0.9 - i * 0.01 for i in range(6)}
    scores.update({"100": 0.8, "101": 0.7})
    identity_rerank = _fixed_score_rerank(scores)

    result = retrieve_rung5(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
        rerank_fn=identity_rerank,
    )

    fiche_contexts = [c for c in result.contexts if c.register is Register.FICHE]
    article_contexts = [c for c in result.contexts if c.register is Register.ARTICLE]
    assert len(fiche_contexts) == 4
    assert len(article_contexts) == 2
    assert result.floor_met is True
    assert set(result.candidate_pools) == {FICHE_LEG, ARTICLE_LEG, EXPANSION_POOL}


def test_rung5_article_floor_not_met_inserts_the_no_article_marker(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1"})])
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(2, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2"})],
    )
    below_floor_rerank = _fixed_score_rerank({"1": 0.9, "2": 0.1})

    result = retrieve_rung5(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
        rerank_fn=below_floor_rerank,
    )

    assert result.floor_met is False
    article_contexts = [c for c in result.contexts if c.register is Register.ARTICLE]
    [marker] = article_contexts
    assert marker.provenance == frozenset()


def test_rung5_quota_and_floor_are_overridable_by_the_caller(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """This ticket's own acceptance criterion: quota depth and the floor value are config,
    not code — the rung-5-vs-rung-1 comparison stays a config change."""
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1"})])
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(2, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2"})],
    )
    low_score_rerank = _fixed_score_rerank({"1": 0.9, "2": 0.2})

    result = retrieve_rung5(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
        rerank_fn=low_score_rerank,
        relevance_floor=0.1,
    )

    assert result.floor_met is True
    article_contexts = [c for c in result.contexts if c.register is Register.ARTICLE]
    assert [c.id for c in article_contexts] == ["2"]


def test_rung5_rerank_fn_defaults_to_none_and_resolves_the_real_reranker(
    monkeypatch: pytest.MonkeyPatch, qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """Mirrors rung 4's own equivalent test: with no `rerank_fn` injected, `retrieve_rung5`
    reaches `rag.retrieval.rerank.rerank` bound to `reranker_model`/`reranker_backend`."""
    import rag.retrieval.pipeline as pipeline_module
    from rag.retrieval.rerank import RerankerBackend

    seen: dict[str, object] = {}

    def _fake_rerank(
        query: str, candidates: list[Candidate], *, model_id: str, backend: RerankerBackend
    ) -> list[Candidate]:
        seen["query"] = query
        seen["model_id"] = model_id
        seen["backend"] = backend
        return candidates

    monkeypatch.setattr(pipeline_module, "rerank", _fake_rerank)
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)

    retrieve_rung5(
        qdrant,
        stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        "une question ouverte",
        set(),
        reranker_model="Alibaba-NLP/gte-multilingual-reranker-base",
        reranker_backend=RerankerBackend.ONNX_INT8,
    )

    assert seen["query"] == "une question ouverte"
    assert seen["model_id"] == "Alibaba-NLP/gte-multilingual-reranker-base"
    assert seen["backend"] is RerankerBackend.ONNX_INT8


def test_rung5_is_registered_but_rung1_stays_the_default_arm() -> None:
    assert RETRIEVAL_ARMS["rung5"] is retrieve_rung5
    assert DEFAULT_RETRIEVAL_ARM == "rung1"
    assert RETRIEVAL_ARMS[DEFAULT_RETRIEVAL_ARM] is retrieve_rung1
