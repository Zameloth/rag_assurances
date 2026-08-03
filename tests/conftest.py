"""Fixtures shared across the suite."""

import os
from collections.abc import Iterator

import pytest
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rag.config import DEFAULT_QDRANT_URL


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
    # `.env` rather than the default alone, so a dev who moved the store is followed.
    load_dotenv()
    url = os.environ.get("QDRANT_URL") or DEFAULT_QDRANT_URL
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
