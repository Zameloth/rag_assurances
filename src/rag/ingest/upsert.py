"""Idempotent upsert orchestration (SPEC §6.4, §7, #25).

Ties the chunker, the payload builder and a caller-supplied embedder together into points,
then upserts them. Idempotency needs nothing extra here — it falls straight out of
`payload.py`'s UUIDv5 point ids: the same row chunked the same way always yields the same
ids, so a second run overwrites in place rather than duplicating.

`embed` takes the whole batch of chunk texts at once and returns dense+sparse pairs in the
same order, one per text — the natural shape for BGE-M3's one-forward-pass-per-batch
encoding (#26), and the seam #25 left for that model to plug into.

**`dense_text` is the embedding-time enrichment seam (SPEC §12.8, #38).** `None` (every
default arm) keeps the one-forward-pass-per-batch shape: `embed` is called once, on the raw
chunk texts, and its dense and sparse halves are both kept. A challenger arm passes a
function instead — `rag.ingest.enrichment.enrich_article_dense_text`/
`enrich_fiche_dense_text` — and `embed` is called *twice*: once on the enriched texts (dense
half kept, sparse half discarded) and once on the raw texts (sparse half kept, dense half
discarded). The split is why two calls, not one: BGE-M3's single forward pass emits dense
and sparse from the *same* input text, and the two pre-ladder A/Bs need those to differ. The
raw chunk text — never the enriched text — is what `build_article_point`/`build_fiche_point`
write into the payload either way (SPEC §7: "`text` is always the raw chunk").

Upserting is chunked into `_UPSERT_BATCH_SIZE`-point requests, not one call per collection:
a single 1024-dim-fp32-dense-plus-sparse point runs ~25 KB of JSON, so the full `articles`
collection (2,801 points) serializes past Qdrant's 32 MB REST request cap in one shot
(measured failure: 70 MB, real corpus, #26). 500 points keeps every request under ~12.5 MB —
comfortable margin without adding a network round trip per article.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from qdrant_client import QdrantClient, models

from rag.ingest.articles import ArticleChunk, ArticleRow, chunk_article
from rag.ingest.fiches import FicheChunk, FicheMetadata, chunk_fiche, parse_fiche_metadata
from rag.ingest.payload import build_article_point, build_fiche_point

__all__ = [
    "ArticleDenseTextFn",
    "Embedding",
    "EmbedFn",
    "FicheDenseTextFn",
    "upsert_articles",
    "upsert_fiches",
]

Embedding = tuple[list[float], models.SparseVector]
EmbedFn = Callable[[Sequence[str]], list[Embedding]]
ArticleDenseTextFn = Callable[[ArticleRow, ArticleChunk], str]
FicheDenseTextFn = Callable[[FicheMetadata, FicheChunk], str]

# See the module docstring — sized against the measured ~25 KB/point of a real BGE-M3
# point, not the tiny stub vectors the test suite embeds.
_UPSERT_BATCH_SIZE = 500


def upsert_articles(
    client: QdrantClient,
    collection_name: str,
    rows: Iterable[ArticleRow],
    embed: EmbedFn,
    *,
    dense_text: ArticleDenseTextFn | None = None,
) -> int:
    """Chunk every row, embed every chunk in one batch, upsert every point. Returns the
    point count written — `rows` is consumed once, same contract as the assertions module."""
    chunked = [(row, chunk_article(row)) for row in rows]
    flat = [(row, chunk) for row, chunks in chunked for chunk in chunks]
    raw_texts = [chunk.text for _, chunk in flat]
    if dense_text is None:
        embeddings = embed(raw_texts)
    else:
        enriched_texts = [dense_text(row, chunk) for row, chunk in flat]
        dense_only = embed(enriched_texts)
        sparse_only = embed(raw_texts)
        embeddings = [(dense, sparse) for (dense, _), (_, sparse) in zip(dense_only, sparse_only, strict=True)]
    points = [
        build_article_point(row, chunk, dense, sparse)
        for (row, chunk), (dense, sparse) in zip(flat, embeddings, strict=True)
    ]
    _upsert_in_batches(client, collection_name, points)
    return len(points)


def upsert_fiches(
    client: QdrantClient,
    collection_name: str,
    fiches: Iterable[bytes],
    embed: EmbedFn,
    *,
    dense_text: FicheDenseTextFn | None = None,
) -> int:
    """The fiche analogue of `upsert_articles`. `fiches` is raw DILA XML bytes, one per
    fiche — metadata and chunks are both derived from it here."""
    parsed = [(parse_fiche_metadata(xml), chunk_fiche(xml)) for xml in fiches]
    flat = [(meta, chunk) for meta, chunks in parsed for chunk in chunks]
    raw_texts = [chunk.text for _, chunk in flat]
    if dense_text is None:
        embeddings = embed(raw_texts)
    else:
        enriched_texts = [dense_text(meta, chunk) for meta, chunk in flat]
        dense_only = embed(enriched_texts)
        sparse_only = embed(raw_texts)
        embeddings = [(dense, sparse) for (dense, _), (_, sparse) in zip(dense_only, sparse_only, strict=True)]
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
