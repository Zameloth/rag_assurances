"""The dev store from docker-compose.yml, when it is up (#25).

Skipped otherwise — see the `qdrant_server` fixture. What is asserted here is what local
mode cannot vouch for (see `test_arms.py`'s module docstring): `indexing_threshold`
actually holding at zero, and the two payload indexes actually taking effect.
"""

from conftest import drop_collections
from qdrant_client import QdrantClient, models

from rag.ingest.arms import ensure_articles_collection, ensure_fiches_collection

DENSE_DIM = 4


def test_the_engine_holds_articles_indexing_threshold_at_zero(qdrant_server: QdrantClient) -> None:
    try:
        ensure_articles_collection(qdrant_server, "pytest__articles_ensure", dense_dim=DENSE_DIM)

        info = qdrant_server.get_collection("pytest__articles_ensure")

        assert info.config.optimizer_config.indexing_threshold == 0
    finally:
        drop_collections(qdrant_server, "pytest__articles_ensure")


def test_the_engine_holds_fiches_indexing_threshold_at_zero(qdrant_server: QdrantClient) -> None:
    try:
        ensure_fiches_collection(qdrant_server, "pytest__fiches_ensure", dense_dim=DENSE_DIM)

        info = qdrant_server.get_collection("pytest__fiches_ensure")

        assert info.config.optimizer_config.indexing_threshold == 0
    finally:
        drop_collections(qdrant_server, "pytest__fiches_ensure")


def test_articles_collection_has_exactly_two_keyword_payload_indexes(qdrant_server: QdrantClient) -> None:
    """SPEC §7.2 — exactly two payload indexes, both keyword; everything else is a full scan."""
    try:
        ensure_articles_collection(qdrant_server, "pytest__articles_indexes", dense_dim=DENSE_DIM)

        schema = qdrant_server.get_collection("pytest__articles_indexes").payload_schema

        assert set(schema) == {"lookup_key", "section_id"}
        assert all(field.data_type == models.PayloadSchemaType.KEYWORD for field in schema.values())
    finally:
        drop_collections(qdrant_server, "pytest__articles_indexes")


def test_fiches_collection_has_no_payload_indexes(qdrant_server: QdrantClient) -> None:
    """SPEC §7.1 — no indexed field at all; `section_ids` is read, never filtered on."""
    try:
        ensure_fiches_collection(qdrant_server, "pytest__fiches_indexes", dense_dim=DENSE_DIM)

        schema = qdrant_server.get_collection("pytest__fiches_indexes").payload_schema

        assert schema == {}
    finally:
        drop_collections(qdrant_server, "pytest__fiches_indexes")
