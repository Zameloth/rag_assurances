"""SPEC §7.5 — `Candidate`'s provenance-union merge rule."""

from rag.retrieval.candidates import Candidate, Provenance, Register, merge_candidates


def _candidate(id_: str, score: float, provenance: frozenset[Provenance]) -> Candidate:
    return Candidate(
        id=id_, score=score, register=Register.ARTICLE, payload={}, provenance=provenance
    )


def test_merge_is_a_no_op_on_disjoint_ids() -> None:
    fiche_pool = [_candidate("a", 0.9, frozenset({Provenance.SEARCH}))]
    article_pool = [_candidate("b", 0.8, frozenset({Provenance.SEARCH}))]

    merged = merge_candidates(fiche_pool, article_pool)

    assert {c.id for c in merged} == {"a", "b"}


def test_merge_dedupes_by_id_and_unions_provenance() -> None:
    """SPEC §7.5: an article reached by both search and expansion keeps one candidate with
    both provenance labels, not two competing candidates."""
    search_pool = [_candidate("a", 0.5, frozenset({Provenance.SEARCH}))]
    expansion_pool = [_candidate("a", 0.5, frozenset({Provenance.EXPANSION}))]

    merged = merge_candidates(search_pool, expansion_pool)

    assert len(merged) == 1
    assert merged[0].provenance == frozenset({Provenance.SEARCH, Provenance.EXPANSION})


def test_merge_keeps_the_first_pools_score_and_payload_on_a_collision() -> None:
    """Not exercised by rung 1 (disjoint collections per leg), but pins the rule down for
    when expansion (#30) makes collisions real."""
    first = _candidate("a", 0.9, frozenset({Provenance.SEARCH}))
    second = _candidate("a", 0.1, frozenset({Provenance.EXPANSION}))

    merged = merge_candidates([first], [second])

    assert merged[0].score == 0.9


def test_merge_of_no_pools_is_empty() -> None:
    assert merge_candidates() == []
