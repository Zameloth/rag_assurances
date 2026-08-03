"""The dev store from docker-compose.yml, when it is up.

Skipped otherwise — see the `qdrant_server` fixture. What is asserted here is what local
mode cannot vouch for: that the *engine* accepts the SPEC §6.3–§6.4 collection shape.
"""

from qdrant_client import QdrantClient, models

from conftest import CreateCollection, MakePoint

COLLECTION = "pytest__skeleton_probe"
ALIAS = "pytest__skeleton_alias"


def test_the_engine_holds_the_indexing_threshold_at_zero(
    qdrant_server: QdrantClient, create_collection: CreateCollection
) -> None:
    """SPEC §6.3 — without this, collection size alone decides the search algorithm.

    Asserted against the engine because the override is per collection: a server-wide
    default would make this pass while leaving a collection created without it exposed.
    """
    create_collection(qdrant_server, COLLECTION)

    info = qdrant_server.get_collection(COLLECTION)

    assert info.config.optimizer_config.indexing_threshold == 0


def test_the_engine_accepts_two_named_vectors_on_one_point(
    qdrant_server: QdrantClient, create_collection: CreateCollection, make_point: MakePoint
) -> None:
    create_collection(qdrant_server, COLLECTION)

    qdrant_server.upsert(
        collection_name=COLLECTION,
        points=[make_point("LEGIARTI000006794088", 0, [1.0, 0.0, 0.0, 0.0])],
        wait=True,
    )

    assert qdrant_server.count(COLLECTION).count == 1


def test_an_alias_is_what_the_retriever_would_read_through(
    qdrant_server: QdrantClient, create_collection: CreateCollection, make_point: MakePoint
) -> None:
    """SPEC §6.4 — arms are real collections behind stable aliases, so switching is a flip."""
    create_collection(qdrant_server, COLLECTION)
    qdrant_server.upsert(
        collection_name=COLLECTION,
        points=[make_point("LEGIARTI000006794088", 0, [1.0, 0.0, 0.0, 0.0])],
        wait=True,
    )
    qdrant_server.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=COLLECTION, alias_name=ALIAS)
            )
        ]
    )

    try:
        assert qdrant_server.count(ALIAS).count == 1
    finally:
        qdrant_server.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=ALIAS))
            ]
        )
