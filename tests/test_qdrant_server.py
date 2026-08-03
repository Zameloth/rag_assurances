"""The dev store from docker-compose.yml, when it is up.

Skipped otherwise — see the `qdrant_server` fixture. What is asserted here is what local
mode cannot vouch for: that the *engine* accepts the SPEC §6.3–§6.4 collection shape.
"""

from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient, models

COLLECTION = "pytest__skeleton_probe"


@pytest.fixture
def probe_collection(qdrant_server: QdrantClient) -> Iterator[QdrantClient]:
    qdrant_server.delete_collection(COLLECTION)
    qdrant_server.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )
    try:
        yield qdrant_server
    finally:
        qdrant_server.delete_collection(COLLECTION)


def test_the_engine_holds_the_indexing_threshold_at_zero(probe_collection: QdrantClient) -> None:
    """SPEC §6.3 — without this, collection size alone decides the search algorithm."""
    info = probe_collection.get_collection(COLLECTION)

    assert info.config.optimizer_config.indexing_threshold == 0


def test_the_engine_accepts_two_named_vectors_on_one_point(
    probe_collection: QdrantClient,
) -> None:
    probe_collection.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id="0f7f2a4e-1c3b-5d6e-8a9b-0c1d2e3f4a5b",
                vector={
                    "dense": [1.0, 0.0, 0.0, 0.0],
                    "sparse": models.SparseVector(indices=[1, 7], values=[0.9, 0.4]),
                },
                payload={"register": "articles"},
            )
        ],
        wait=True,
    )

    assert probe_collection.count(COLLECTION).count == 1


def test_an_alias_is_what_the_retriever_would_read_through(
    probe_collection: QdrantClient,
) -> None:
    """SPEC §6.4 — arms are real collections behind stable aliases, so switching is a flip."""
    probe_collection.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=COLLECTION, alias_name="pytest__skeleton_alias"
                )
            )
        ]
    )
    try:
        assert probe_collection.count("pytest__skeleton_alias").count == 0
    finally:
        probe_collection.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name="pytest__skeleton_alias")
                )
            ]
        )
