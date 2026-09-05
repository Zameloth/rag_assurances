"""SPEC §6.3 / §6.4 / §7.2 / ADR-0005 / #25 — arm collections, indexes, alias flip.

Collection *shape* (vectors, `indexing_threshold=0`, idempotent re-creation, alias flip) is
plumbing the in-memory client can vouch for. Whether a payload index actually *takes
effect* is not — local mode warns "Payload indexes have no effect in the local Qdrant" and
leaves `payload_schema` empty regardless — so that one claim is asserted against the real
engine in `test_arms_server.py`, skipped when it is not up, the same split
`test_qdrant_fixture.py` / `test_qdrant_server.py` already make.
"""

from conftest import drop_collections
from qdrant_client import QdrantClient, models

from rag.ingest.arms import ensure_articles_collection, ensure_fiches_collection, flip_alias

DENSE_DIM = 4


def _assert_has_dense_and_sparse(info: models.CollectionInfo) -> None:
    vectors, sparse_vectors = info.config.params.vectors, info.config.params.sparse_vectors
    assert isinstance(vectors, dict) and "dense" in vectors
    assert isinstance(sparse_vectors, dict) and "sparse" in sparse_vectors


def test_ensure_articles_collection_creates_dense_and_sparse_vectors(qdrant: QdrantClient) -> None:
    try:
        ensure_articles_collection(qdrant, "articles__test", dense_dim=DENSE_DIM)

        _assert_has_dense_and_sparse(qdrant.get_collection("articles__test"))
    finally:
        drop_collections(qdrant, "articles__test")


def test_ensure_fiches_collection_creates_dense_and_sparse_vectors(qdrant: QdrantClient) -> None:
    try:
        ensure_fiches_collection(qdrant, "fiches__test", dense_dim=DENSE_DIM)

        _assert_has_dense_and_sparse(qdrant.get_collection("fiches__test"))
    finally:
        drop_collections(qdrant, "fiches__test")


def test_ensure_articles_collection_is_idempotent(qdrant: QdrantClient) -> None:
    """Re-running the ingest must not fail on a collection that already exists."""
    try:
        ensure_articles_collection(qdrant, "articles__test", dense_dim=DENSE_DIM)
        ensure_articles_collection(qdrant, "articles__test", dense_dim=DENSE_DIM)

        assert qdrant.collection_exists("articles__test")
    finally:
        drop_collections(qdrant, "articles__test")


def test_ensure_fiches_collection_is_idempotent(qdrant: QdrantClient) -> None:
    try:
        ensure_fiches_collection(qdrant, "fiches__test", dense_dim=DENSE_DIM)
        ensure_fiches_collection(qdrant, "fiches__test", dense_dim=DENSE_DIM)

        assert qdrant.collection_exists("fiches__test")
    finally:
        drop_collections(qdrant, "fiches__test")


def test_flip_alias_points_the_stable_name_at_a_physical_collection(qdrant: QdrantClient) -> None:
    """SPEC §6.4 — no experiment or application code ever names a physical collection."""
    try:
        ensure_articles_collection(qdrant, "articles__arm1", dense_dim=DENSE_DIM)
        qdrant.upsert(
            collection_name="articles__arm1",
            points=[
                models.PointStruct(
                    id=1,
                    vector={"dense": [1.0, 0.0, 0.0, 0.0], "sparse": models.SparseVector(indices=[1], values=[1.0])},
                    payload={},
                )
            ],
        )

        flip_alias(qdrant, "articles", "articles__arm1")

        assert qdrant.count("articles").count == 1
        aliases = {a.alias_name: a.collection_name for a in qdrant.get_aliases().aliases}
        assert aliases["articles"] == "articles__arm1"
    finally:
        drop_collections(qdrant, "articles__arm1")


def test_flip_alias_switching_arms_is_atomic_and_leaves_no_stale_binding(qdrant: QdrantClient) -> None:
    """Switching arms is an alias update — rolling back is instant, and the old physical
    collection must not remain reachable under the stable name."""
    try:
        ensure_articles_collection(qdrant, "articles__arm1", dense_dim=DENSE_DIM)
        ensure_articles_collection(qdrant, "articles__arm2", dense_dim=DENSE_DIM)
        flip_alias(qdrant, "articles", "articles__arm1")

        flip_alias(qdrant, "articles", "articles__arm2")

        aliases = [a for a in qdrant.get_aliases().aliases if a.alias_name == "articles"]
        assert len(aliases) == 1
        assert aliases[0].collection_name == "articles__arm2"
    finally:
        drop_collections(qdrant, "articles__arm1", "articles__arm2")


def test_flip_alias_onto_the_same_arm_is_a_no_op(qdrant: QdrantClient) -> None:
    try:
        ensure_articles_collection(qdrant, "articles__arm1", dense_dim=DENSE_DIM)
        flip_alias(qdrant, "articles", "articles__arm1")

        flip_alias(qdrant, "articles", "articles__arm1")

        aliases = [a for a in qdrant.get_aliases().aliases if a.alias_name == "articles"]
        assert len(aliases) == 1
    finally:
        drop_collections(qdrant, "articles__arm1")
