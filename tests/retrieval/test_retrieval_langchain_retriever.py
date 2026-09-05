"""SPEC §6.1, #28 — the `BaseRetriever` wrapper: native `qdrant-client` reads through
`rag.retrieval.pipeline.retrieve`, `Candidate` -> `Document` mapping (register/provenance
as metadata, never stored)."""

from conftest import CreateCollection, raw_point, stub_embed
from qdrant_client import QdrantClient

from rag.retrieval.langchain_retriever import PipelineRetriever
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS


def test_invoke_maps_a_candidate_to_a_document_with_register_and_provenance_metadata(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2", "text": "Le texte."})],
    )
    retriever = PipelineRetriever(
        client=qdrant, embed=stub_embed([1.0, 0.0, 0.0, 0.0]), lookup_keys=set()
    )

    documents = retriever.invoke("Quelle franchise pour un dégât des eaux ?")

    [document] = documents
    assert document.id == "1"
    assert document.page_content == "Le texte."
    assert document.metadata == {"register": "article", "provenance": ["search"]}


def test_invoke_resolves_the_short_circuit_and_tags_provenance_lookup(
    qdrant: QdrantClient, create_collection: CreateCollection
) -> None:
    """Path selection (SPEC §9.1) runs before rung 1 ever does — a membership hit skips
    search entirely, so only `ARTICLES_ALIAS` needs to exist, same as
    `test_a_resolved_short_circuit_skips_search_and_has_no_candidate_pools` in
    `test_retrieval_pipeline.py`. This exercises the same path *through* the retriever, to
    confirm the `Candidate` -> `Document` mapping also holds for `Provenance.LOOKUP`."""
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[
            raw_point(
                1, [0.0, 0.0, 0.0, 1.0], {"lookup_key": "L113-2", "chunk_index": 0, "text": "Le texte."}
            )
        ],
    )
    retriever = PipelineRetriever(
        client=qdrant, embed=stub_embed([1.0, 0.0, 0.0, 0.0]), lookup_keys={"L113-2"}
    )

    documents = retriever.invoke("Que dit L113-2 sur la résiliation ?")

    [document] = documents
    assert document.page_content == "Le texte."
    assert document.metadata == {"register": "article", "provenance": ["lookup"]}
