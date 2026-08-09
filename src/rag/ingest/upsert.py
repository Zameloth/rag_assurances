"""Idempotent upsert orchestration (SPEC §6.4, §7, #25).

Ties the chunker, the payload builder and a caller-supplied embedder together into points,
then upserts them. Idempotency needs nothing extra here — it falls straight out of
`payload.py`'s UUIDv5 point ids: the same row chunked the same way always yields the same
ids, so a second run overwrites in place rather than duplicating.

`embed` takes the whole batch of chunk texts at once and returns dense+sparse pairs in the
same order, one per text — the natural shape for BGE-M3's one-forward-pass-per-batch
encoding (#26), and the seam #25 left for that model to plug into.

Upserting is chunked into `_UPSERT_BATCH_SIZE`-point requests, not one call per collection:
a single 1024-dim-fp32-dense-plus-sparse point runs ~25 KB of JSON, so the full `articles`
collection (2,801 points) serializes past Qdrant's 32 MB REST request cap in one shot
(measured failure: 70 MB, real corpus, #26). 500 points keeps every request under ~12.5 MB —
comfortable margin without adding a network round trip per article.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from qdrant_client import QdrantClient, models

from rag.ingest.articles import ArticleRow, chunk_article
from rag.ingest.fiches import chunk_fiche, parse_fiche_metadata
from rag.ingest.payload import build_article_point, build_fiche_point

__all__ = ["Embedding", "EmbedFn", "upsert_articles", "upsert_fiches"]

Embedding = tuple[list[float], models.SparseVector]
EmbedFn = Callable[[Sequence[str]], list[Embedding]]

# See the module docstring — sized against the measured ~25 KB/point of a real BGE-M3
# point, not the tiny stub vectors the test suite embeds.
_UPSERT_BATCH_SIZE = 500


def upsert_articles(
    client: QdrantClient, collection_name: str, rows: Iterable[ArticleRow], embed: EmbedFn
) -> int:
    """Chunk every row, embed every chunk in one batch, upsert every point. Returns the
    point count written — `rows` is consumed once, same contract as the assertions module."""
    chunked = [(row, chunk_article(row)) for row in rows]
    flat = [(row, chunk) for row, chunks in chunked for chunk in chunks]
    embeddings = embed([chunk.text for _, chunk in flat])
    points = [
        build_article_point(row, chunk, dense, sparse)
        for (row, chunk), (dense, sparse) in zip(flat, embeddings, strict=True)
    ]
    _upsert_in_batches(client, collection_name, points)
    return len(points)


def upsert_fiches(
    client: QdrantClient, collection_name: str, fiches: Iterable[bytes], embed: EmbedFn
) -> int:
    """The fiche analogue of `upsert_articles`. `fiches` is raw DILA XML bytes, one per
    fiche — metadata and chunks are both derived from it here."""
    parsed = [(parse_fiche_metadata(xml), chunk_fiche(xml)) for xml in fiches]
    flat = [(meta, chunk) for meta, chunks in parsed for chunk in chunks]
    embeddings = embed([chunk.text for _, chunk in flat])
    points = [
        build_fiche_point(meta, chunk, dense, sparse)
        for (meta, chunk), (dense, sparse) in zip(flat, embeddings, strict=True)
    ]
    _upsert_in_batches(client, collection_name, points)
    return len(points)


def _upsert_in_batches(
    client: QdrantClient, collection_name: str, points: list[models.PointStruct]
) -> None:
    for start in range(0, len(points), _UPSERT_BATCH_SIZE):
        batch = points[start : start + _UPSERT_BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
