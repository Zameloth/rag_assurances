"""The plumbing later tickets build on, exercised against the in-memory client.

These assertions are about the store accepting the shape SPEC §6.3–§6.4 fixes — exact
search, two named vectors on one point, UUIDv5 ids — not about retrieval quality.
"""

from qdrant_client import QdrantClient, models

from conftest import CreateCollection, MakePoint


def test_the_fixture_accepts_dense_and_sparse_as_two_named_vectors(
    qdrant: QdrantClient, create_collection: CreateCollection, make_point: MakePoint
) -> None:
    create_collection(qdrant, "articles")

    qdrant.upsert(
        collection_name="articles",
        points=[make_point("LEGIARTI000006794088", 0, [1.0, 0.0, 0.0, 0.0])],
    )

    assert qdrant.count("articles").count == 1


def test_a_dense_query_returns_the_payload_verbatim(
    qdrant: QdrantClient, create_collection: CreateCollection, make_point: MakePoint
) -> None:
    create_collection(qdrant, "articles")
    qdrant.upsert(
        collection_name="articles",
        points=[
            make_point("LEGIARTI000006794088", 0, [1.0, 0.0, 0.0, 0.0]),
            make_point("LEGIARTI000006794089", 0, [0.0, 1.0, 0.0, 0.0]),
        ],
    )

    hits = qdrant.query_points(
        collection_name="articles",
        query=[1.0, 0.0, 0.0, 0.0],
        using="dense",
        limit=1,
    ).points

    assert hits[0].payload == {
        "legiarti_cid": "LEGIARTI000006794088",
        "chunk_index": 0,
        "citation_id": "L113-3",
    }


def test_a_sparse_query_runs_on_raw_indices_and_values(
    qdrant: QdrantClient, create_collection: CreateCollection, make_point: MakePoint
) -> None:
    """SPEC §6.2 — the store must accept M3's learned weights, not generate its own."""
    create_collection(qdrant, "articles")
    qdrant.upsert(
        collection_name="articles",
        points=[make_point("LEGIARTI000006794088", 0, [1.0, 0.0, 0.0, 0.0])],
    )

    hits = qdrant.query_points(
        collection_name="articles",
        query=models.SparseVector(indices=[1], values=[1.0]),
        using="sparse",
        limit=1,
    ).points

    assert len(hits) == 1


def test_reingesting_the_same_chunk_overwrites_rather_than_duplicates(
    qdrant: QdrantClient, create_collection: CreateCollection, make_point: MakePoint
) -> None:
    """SPEC §6.4 — UUIDv5 over the natural key is what makes ingestion idempotent."""
    create_collection(qdrant, "articles")

    for _ in range(2):
        qdrant.upsert(
            collection_name="articles",
            points=[make_point("LEGIARTI000006794088", 0, [1.0, 0.0, 0.0, 0.0])],
        )

    assert qdrant.count("articles").count == 1


def test_a_second_chunk_of_the_same_article_is_its_own_point(
    qdrant: QdrantClient, create_collection: CreateCollection, make_point: MakePoint
) -> None:
    """SPEC §6.4 — 289 articles produce 2+ points, so chunk_index is load-bearing."""
    create_collection(qdrant, "articles")

    qdrant.upsert(
        collection_name="articles",
        points=[
            make_point("LEGIARTI000006794088", 0, [1.0, 0.0, 0.0, 0.0]),
            make_point("LEGIARTI000006794088", 1, [0.0, 1.0, 0.0, 0.0]),
        ],
    )

    assert qdrant.count("articles").count == 2


def test_the_two_registers_are_separate_collections(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """SPEC §6.4 — the topology is two collections, never one with a register field."""
    create_collection(qdrant, "fiches")
    create_collection(qdrant, "articles")

    names = {c.name for c in qdrant.get_collections().collections}

    assert names == {"fiches", "articles"}
