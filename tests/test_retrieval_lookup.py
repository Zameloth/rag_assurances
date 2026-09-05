"""SPEC §9.1 — the short-circuit's metadata-lookup path against `articles`."""

from conftest import CreateCollection, raw_point
from qdrant_client import QdrantClient, models

from rag.retrieval.candidates import Provenance, Register
from rag.retrieval.legs import ARTICLES_ALIAS
from rag.retrieval.lookup import _SCROLL_PAGE_SIZE, load_lookup_keys, lookup_article_chunks_by_key


def _article_point(
    point_id: int, lookup_key: str | None, chunk_index: int, citation_id: str
) -> models.PointStruct:
    return raw_point(
        point_id,
        [1.0, 0.0, 0.0, 0.0],
        {"lookup_key": lookup_key, "chunk_index": chunk_index, "citation_id": citation_id},
    )


def test_load_lookup_keys_collects_every_non_null_key(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[
            _article_point(1, "L113-2", 0, "L113-2"),
            _article_point(2, "L113-3", 0, "L113-3"),
            # The 21 prose annexe labels (SPEC §7.3) — null lookup_key, must not appear.
            _article_point(3, None, 0, "Annexe à l'article A121-1"),
        ],
    )

    keys = load_lookup_keys(qdrant)

    assert keys == frozenset({"L113-2", "L113-3"})


def test_load_lookup_keys_paginates_past_one_scroll_page(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    count = _SCROLL_PAGE_SIZE + 44
    points = [_article_point(i, f"L{i}", 0, f"L{i}") for i in range(count)]
    qdrant.upsert(ARTICLES_ALIAS, points=points)

    keys = load_lookup_keys(qdrant)

    assert keys == frozenset(f"L{i}" for i in range(count))


def test_lookup_article_chunks_by_key_returns_every_chunk_ordered(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """An article's chunks all share one `lookup_key` — the short-circuit resolves to the
    whole article, not just its first chunk."""
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[
            _article_point(1, "L113-2", 1, "L113-2"),
            _article_point(2, "L113-2", 0, "L113-2"),
            _article_point(3, "L113-3", 0, "L113-3"),
        ],
    )

    chunks = lookup_article_chunks_by_key(qdrant, "L113-2")

    assert [c.payload["chunk_index"] for c in chunks] == [0, 1]
    assert all(c.register is Register.ARTICLE for c in chunks)
    assert all(c.provenance == frozenset({Provenance.LOOKUP}) for c in chunks)
    assert all(c.score == 1.0 for c in chunks)


def test_lookup_article_chunks_by_key_paginates_past_one_scroll_page(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """A single article whose chunk count exceeds one scroll page must still come back
    whole — nothing measured bounds chunks-per-article below the page size."""
    create_collection(qdrant, ARTICLES_ALIAS)
    count = _SCROLL_PAGE_SIZE + 10
    points = [_article_point(i, "L113-2", i, "L113-2") for i in range(count)]
    qdrant.upsert(ARTICLES_ALIAS, points=points)

    chunks = lookup_article_chunks_by_key(qdrant, "L113-2")

    assert [c.payload["chunk_index"] for c in chunks] == list(range(count))


def test_lookup_article_chunks_by_key_returns_empty_on_no_match(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(ARTICLES_ALIAS, points=[_article_point(1, "L113-2", 0, "L113-2")])

    assert lookup_article_chunks_by_key(qdrant, "L999-9") == []
