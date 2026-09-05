"""SPEC §6.1, §9.2 — `search_leg` issues a native dense `query_points` call per register."""

from conftest import CreateCollection, raw_point
from qdrant_client import QdrantClient

from rag.retrieval.candidates import Provenance, Register
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS, search_leg


def test_search_leg_queries_the_alias_by_name_dense_only(
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

    hits = search_leg(qdrant, Register.ARTICLE, [1.0, 0.0, 0.0, 0.0], limit=10)

    # Membership only — which hit ranks first is a ranking claim about the store's own
    # cosine math, which SPEC §6.3 reserves for the `qdrant_server` fixture, never
    # `:memory:` (see `tests/conftest.py`'s `qdrant` fixture docstring).
    assert {c.payload["citation_id"] for c in hits} == {"L113-2", "L113-3"}


def test_search_leg_attaches_register_and_search_provenance_never_stored_on_the_payload(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """SPEC §7.5 — register/provenance are annotations the caller attaches, not fields the
    payload ever carries."""
    create_collection(qdrant, FICHES_ALIAS)
    qdrant.upsert(FICHES_ALIAS, points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"fiche_id": "F1"})])

    [hit] = search_leg(qdrant, Register.FICHE, [1.0, 0.0, 0.0, 0.0], limit=10)

    assert hit.register is Register.FICHE
    assert hit.provenance == frozenset({Provenance.SEARCH})
    assert "register" not in hit.payload
    assert "provenance" not in hit.payload


def test_search_leg_respects_the_limit(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[
            raw_point(1, [1.0, 0.0, 0.0, 0.0], {}),
            raw_point(2, [0.9, 0.1, 0.0, 0.0], {}),
            raw_point(3, [0.0, 1.0, 0.0, 0.0], {}),
        ],
    )

    hits = search_leg(qdrant, Register.ARTICLE, [1.0, 0.0, 0.0, 0.0], limit=1)

    assert len(hits) == 1
