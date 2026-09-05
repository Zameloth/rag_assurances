"""Arm collections, payload indexes and alias flips (SPEC §6.3, §6.4, §7.2, ADR-0005, #25).

**Ablation arms are real collections behind stable aliases** — `ensure_articles_collection`
/ `ensure_fiches_collection` create the physical arm (`articles__m3__c512__v1` style);
`flip_alias` is the only thing that ever points the stable name (`fiches` / `articles`) at
one. No experiment or application code names a physical collection directly.

`indexing_threshold=0` on every collection (SPEC §6.3) — the two legs of one query must
never run different search algorithms as a side effect of collection size. The two payload
indexes (SPEC §7.2) are `articles`-only: `fiches` has none (SPEC §7.1's index column is
empty end to end), and adding them there would be a full scan Qdrant already runs fast
enough over ~900 points, indexing RAM this project cannot spare for nothing.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

__all__ = [
    "ARTICLES_ALIAS",
    "DENSE_DIM",
    "FICHES_ALIAS",
    "ensure_articles_collection",
    "ensure_fiches_collection",
    "flip_alias",
]

# The two SPEC §6.4 stable names, defined once here — this module is their natural single
# home, since `flip_alias` is the only thing that ever points either at a physical
# collection. `rag.ingest.pipeline` (which flips them) and `rag.retrieval.legs` (which
# queries through them) both import these rather than restating the literals.
ARTICLES_ALIAS = "articles"
FICHES_ALIAS = "fiches"

# BGE-M3's dense output (SPEC §5). Collection-creation callers override it for tests, where
# the point is collection *shape*, not the real embedder.
DENSE_DIM = 1024


def ensure_articles_collection(
    client: QdrantClient, name: str, *, dense_dim: int = DENSE_DIM
) -> None:
    """Create the `articles` arm `name` if it does not already exist, with its two payload
    indexes (SPEC §7.2: `lookup_key` for the short-circuit, `section_id` for expansion,
    both keyword). A no-op on a name that already exists — re-running the ingest against an
    established arm must not fail on this step."""
    if client.collection_exists(name):
        return
    _create_collection(client, name, dense_dim)
    field_schema = models.PayloadSchemaType.KEYWORD
    client.create_payload_index(name, field_name="lookup_key", field_schema=field_schema)
    client.create_payload_index(name, field_name="section_id", field_schema=field_schema)


def ensure_fiches_collection(
    client: QdrantClient, name: str, *, dense_dim: int = DENSE_DIM
) -> None:
    """Create the `fiches` arm `name` if it does not already exist. No payload indexes —
    SPEC §7.1 carries none; `section_ids` is read by expansion, never filtered on."""
    if client.collection_exists(name):
        return
    _create_collection(client, name, dense_dim)


def _create_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    vectors_config = {"dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)}
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_config,
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
        # SPEC §6.3 — HNSW never builds, so search stays exact regardless of collection size.
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )


def flip_alias(client: QdrantClient, alias: str, collection_name: str) -> None:
    """Point the stable `alias` at `collection_name`, atomically dropping whichever physical
    collection it pointed at before (SPEC §6.4) — so a switch is a single alias update and
    rollback is instant. A no-op if `alias` already points at `collection_name`."""
    current = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    if current.get(alias) == collection_name:
        return
    operations: list[models.AliasOperations] = []
    if alias in current:
        delete_op = models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        operations.append(delete_op)
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection_name, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)
