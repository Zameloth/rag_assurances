"""Fixtures shared across the suite.

The Qdrant shapes live here rather than in either test module because both the in-memory
and the server suite assert against the same collection and point layout — SPEC §6.3–§6.4
— and two copies of it would be two things to keep in step with the spec.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient, models

from rag.config import load_settings
from rag.ingest.payload import article_point_id
from rag.ingest.upsert import Embedding, EmbedFn

CreateCollection = Callable[[QdrantClient, str], None]
MakePoint = Callable[[str, int, list[float]], models.PointStruct]


def raw_point(point_id: int, dense: list[float], payload: Mapping[str, Any]) -> models.PointStruct:
    """A bare point for retrieval-plumbing tests: one dense vector, a fixed 1-index sparse
    component — indices=[1], values=[0.5] on every point, so a collection created with
    `sparse_vectors_config` never rejects the upsert, and rung-2 hybrid tests that do query
    the sparse half get a deterministic dot product rather than an arbitrary one — and
    whatever payload the test needs.

    Unlike `make_point` below, this derives no natural-key id and assumes no fixed payload
    shape — `rag.retrieval`'s tests exercise several different payload shapes (fiche vs
    article, with vs without `lookup_key`) against the same tiny collections."""
    return models.PointStruct(
        id=point_id,
        vector={"dense": dense, "sparse": models.SparseVector(indices=[1], values=[0.5])},
        payload=dict(payload),
    )


def stub_embed(dense: list[float]) -> EmbedFn:
    """An `EmbedFn` returning `dense` (and an empty sparse vector) for every text — the
    rung-1 retriever only reads the dense half of the BGE-M3 embedding shape (SPEC §12.7:
    "single index, dense-only"), so the sparse half's exact content is never asserted on
    and an empty one is exactly as informative as a real one here."""
    return stub_embed_hybrid(dense, models.SparseVector(indices=[], values=[]))


def stub_embed_hybrid(dense: list[float], sparse: models.SparseVector) -> EmbedFn:
    """`stub_embed`'s rung-2 counterpart: both halves are real, for tests exercising
    `retrieve_rung2`'s hybrid legs (SPEC §9.3, #29) rather than rung 1's dense-only path."""

    def embed(texts: Sequence[str]) -> list[Embedding]:
        return [(dense, sparse)] * len(texts)

    return embed


def drop_collections(client: QdrantClient, *names: str) -> None:
    """Shared teardown for tests exercising collection-creation code (`ensure_*_collection`)
    directly rather than through the `create_collection` fixture below, which always creates
    unconditionally and so can't be used to test a function whose job is the creation step."""
    for name in names:
        client.delete_collection(name)


@pytest.fixture
def qdrant() -> Iterator[QdrantClient]:
    """An in-process Qdrant, per SPEC §6.3.

    Local mode is a pure-Python reimplementation rather than the Rust engine, so its
    parity gap is irrelevant here and disqualifying anywhere else: use this fixture for
    plumbing assertions — collection shape, payload round-trips, id determinism — and
    never for recall or ranking numbers.
    """
    client = QdrantClient(":memory:")
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def qdrant_server() -> Iterator[QdrantClient]:
    """The real engine from docker-compose.yml, skipped when it is not up.

    Skipping rather than failing is deliberate: the suite must stay green without a live
    store, so anything that needs the Rust engine — the ranking and recall numbers local
    mode cannot speak to — opts in through this fixture and is simply absent otherwise.
    """
    url = load_settings().qdrant_url
    client = QdrantClient(url, timeout=2)
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001 — any failure to reach it means the same thing
        client.close()
        pytest.skip(f"no Qdrant at {url} ({type(exc).__name__}); run `make up`")
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def create_collection() -> Iterator[CreateCollection]:
    """Create a collection in the SPEC §6.3–§6.4 shape, and drop it afterwards.

    Dropping matters for the server fixture, whose store outlives the test run; the
    in-memory client would have thrown it away regardless.
    """
    created: list[tuple[QdrantClient, str]] = []

    def create(client: QdrantClient, name: str) -> None:
        client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
            # SPEC §6.3 — HNSW never builds, so every collection searches exactly
            # regardless of how many points it holds.
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        created.append((client, name))

    try:
        yield create
    finally:
        for client, name in reversed(created):
            client.delete_collection(name)


_LANGFUSE_TEST_SPAN_EXPORTER = InMemorySpanExporter()
_langfuse_test_client: Langfuse | None = None


@pytest.fixture
def langfuse_span_exporter() -> Iterator[InMemorySpanExporter]:
    """A `Langfuse` client wired to an in-memory OTEL exporter, for tests asserting on
    emitted spans (#28, #31).

    Built once per process, not once per test: OpenTelemetry's `TracerProvider` is a
    process-wide singleton (`set_tracer_provider` is a no-op past the first call), so a
    second `Langfuse(...)` construction would silently keep sending spans to the *first*
    test's exporter instead of its own — verified directly, and it's exactly what broke
    the moment a second Langfuse-observation test (#31's rerank span) joined the one from
    #28.

    Flush-then-clear happens in teardown, not at setup: the SDK's span processor batches
    and exports asynchronously, so a test that raises before calling `.flush()` itself
    (an assertion failure, or #31's still-`NotImplementedError` `_traced` stub) can still
    have spans land in the exporter *after* a plain setup-time `.clear()` would have run —
    forcing the flush here is what actually guarantees the next test starts from empty.
    """
    global _langfuse_test_client
    if _langfuse_test_client is None:
        _langfuse_test_client = Langfuse(
            public_key="test", secret_key="test", span_exporter=_LANGFUSE_TEST_SPAN_EXPORTER
        )
    yield _LANGFUSE_TEST_SPAN_EXPORTER
    _langfuse_test_client.flush()
    _LANGFUSE_TEST_SPAN_EXPORTER.clear()


@pytest.fixture
def langfuse_client(langfuse_span_exporter: InMemorySpanExporter) -> Langfuse:
    """The shared client itself, for tests that call `.flush()` to force span export
    before reading `langfuse_span_exporter.get_finished_spans()`."""
    assert _langfuse_test_client is not None  # langfuse_span_exporter always creates it first
    return _langfuse_test_client


@pytest.fixture
def make_point() -> MakePoint:
    """An `articles` point: UUIDv5 id, two named vectors, flat payload (SPEC §6.4, §7.2).

    The payload carries no `register` — SPEC §7.5 makes it an annotation the retriever
    attaches, and a stored copy can only ever disagree with the collection holding it.
    """

    def build(legiarti_cid: str, chunk_index: int, dense: list[float]) -> models.PointStruct:
        return models.PointStruct(
            id=article_point_id(legiarti_cid, chunk_index),
            vector={
                "dense": dense,
                "sparse": models.SparseVector(indices=[1, 7], values=[0.9, 0.4]),
            },
            payload={
                "legiarti_cid": legiarti_cid,
                "chunk_index": chunk_index,
                "citation_id": "L113-3",
            },
        )

    return build
