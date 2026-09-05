"""`python -m rag.query` — CLI wiring, injected client/embed (SPEC §9, #28).

Never loads BGE-M3: `main`'s `embed` parameter is always supplied, the same injection seam
`rag.ingest.pipeline.main` uses so its own tests never load the real model either.
"""

import pytest
from conftest import CreateCollection, raw_point, stub_embed
from qdrant_client import QdrantClient

from rag.query import main
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS
from rag.retrieval.short_circuit import ShortCircuitPath


def test_main_prints_register_score_and_provenance_for_each_context(
    qdrant: QdrantClient, create_collection: CreateCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2", "text": "Le texte."})],
    )

    result = main(
        ["Quelle franchise pour un dégât des eaux ?"],
        client=qdrant,
        embed=stub_embed([1.0, 0.0, 0.0, 0.0]),
    )

    out = capsys.readouterr().out
    assert result.short_circuit_path is ShortCircuitPath.NO_REFERENCE
    assert "register=article" in out
    assert "score=" in out
    assert "provenance=search" in out
    assert "L113-2" in out


def test_main_short_circuits_and_reports_no_search(
    qdrant: QdrantClient, create_collection: CreateCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"lookup_key": "L113-2", "chunk_index": 0})],
    )

    result = main(
        ["Que dit L113-2 sur la résiliation ?"],
        client=qdrant,
        embed=stub_embed([0.0, 1.0, 0.0, 0.0]),
    )

    out = capsys.readouterr().out
    assert result.short_circuit_path is ShortCircuitPath.RESOLVED
    assert "short-circuit" in out
    assert "provenance=lookup" in out


def test_main_reports_no_contexts_when_nothing_matches(
    qdrant: QdrantClient, create_collection: CreateCollection, capsys: pytest.CaptureFixture[str]
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)

    main(["une question ouverte"], client=qdrant, embed=stub_embed([1.0, 0.0, 0.0, 0.0]))

    out = capsys.readouterr().out
    assert "no contexts returned" in out


def test_main_does_not_close_an_injected_client(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """The CLI only owns (and closes) a client it created itself — an injected one is the
    test's to close, the same contract `rag.ingest.pipeline.main` follows."""
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)

    main(["une question"], client=qdrant, embed=stub_embed([1.0, 0.0, 0.0, 0.0]))

    # Still usable — closing it would make this raise.
    qdrant.get_collections()
