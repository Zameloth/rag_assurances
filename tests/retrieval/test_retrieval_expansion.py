"""SPEC §9.2, ADR-0006, ADR-0017, #30 — `<dc:source>` expansion, the rung-3 arm's spine."""

from conftest import CreateCollection, raw_point
from qdrant_client import QdrantClient

from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.expansion import (
    EXPANSION_CAP,
    EXPANSION_FICHE_DEPTH,
    expand,
    expand_by_section,
    section_ids_from_fiches,
    top_fiches,
)
from rag.retrieval.legs import ARTICLES_ALIAS


def _fiche(id_: str, score: float, section_ids: list[str] | None = None) -> Candidate:
    payload = {} if section_ids is None else {"section_ids": section_ids}
    return Candidate(
        id=id_,
        score=score,
        register=Register.FICHE,
        payload=payload,
        provenance=frozenset({Provenance.SEARCH}),
    )


def test_top_fiches_takes_the_first_depth_elements_positionally() -> None:
    """No re-sort — `fiche_pool` is assumed already ranked by score descending, the shape
    `hybrid_leg`/`search_leg` return (ADR-0017)."""
    pool = [_fiche("a", 0.9), _fiche("b", 0.5), _fiche("c", 0.1)]

    assert [c.id for c in top_fiches(pool, depth=2)] == ["a", "b"]


def test_top_fiches_depth_beyond_pool_length_returns_the_whole_pool() -> None:
    pool = [_fiche("a", 0.9)]

    assert [c.id for c in top_fiches(pool, depth=3)] == ["a"]


def test_section_ids_from_fiches_unions_and_dedupes_preserving_first_seen_order() -> None:
    fiches = [
        _fiche("a", 0.9, ["S1", "S2"]),
        _fiche("b", 0.5, ["S2", "S3"]),
    ]

    assert section_ids_from_fiches(fiches) == ["S1", "S2", "S3"]


def test_section_ids_from_fiches_treats_missing_section_ids_as_empty() -> None:
    """A fiche chunk with no `section_ids` key contributes nothing rather than erroring —
    SPEC §7.1 is read-only here, never assumed present."""
    fiches = [_fiche("a", 0.9, section_ids=None), _fiche("b", 0.5, ["S1"])]

    assert section_ids_from_fiches(fiches) == ["S1"]


def test_expand_by_section_returns_nothing_for_an_empty_section_id_list(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """Never a filterless query in disguise (ADR-0017) — an empty `section_ids` list must
    not silently degrade into an unfiltered top-`cap` search."""
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(ARTICLES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"section_id": "S1"})])

    assert expand_by_section(qdrant, [1.0, 0.0, 0.0, 0.0], []) == []


def test_expand_by_section_filters_by_match_any_on_section_id(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[
            raw_point(1, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L1", "section_id": "S1"}),
            raw_point(2, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L2", "section_id": "S2"}),
            raw_point(3, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L3", "section_id": "S3"}),
        ],
    )

    pool = expand_by_section(qdrant, [1.0, 0.0, 0.0, 0.0], ["S1", "S2"])

    assert {c.payload["citation_id"] for c in pool} == {"L1", "L2"}


def test_expand_by_section_attaches_expansion_provenance_and_article_register(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(ARTICLES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"section_id": "S1"})])

    [hit] = expand_by_section(qdrant, [1.0, 0.0, 0.0, 0.0], ["S1"])

    assert hit.provenance == frozenset({Provenance.EXPANSION})
    assert hit.register is Register.ARTICLE


def test_expand_by_section_respects_the_cap(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(i, [1.0, 0.0, 0.0, 0.0], {"section_id": "S1"}) for i in range(5)],
    )

    pool = expand_by_section(qdrant, [1.0, 0.0, 0.0, 0.0], ["S1"], cap=2)

    assert len(pool) == 2


def test_expand_reads_section_ids_from_only_the_top_depth_fiches(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[
            raw_point(1, [1.0, 0.0, 0.0, 0.0], {"citation_id": "in-window", "section_id": "S1"}),
            raw_point(2, [1.0, 0.0, 0.0, 0.0], {"citation_id": "out-of-window", "section_id": "S9"}),
        ],
    )
    fiche_pool = [_fiche("a", 0.9, ["S1"]), _fiche("b", 0.1, ["S9"])]

    pool = expand(qdrant, fiche_pool, [1.0, 0.0, 0.0, 0.0], depth=1)

    assert {c.payload["citation_id"] for c in pool} == {"in-window"}


def test_expansion_fiche_depth_and_cap_constants_match_spec_9_2() -> None:
    assert EXPANSION_FICHE_DEPTH == 3
    assert EXPANSION_CAP == 40
