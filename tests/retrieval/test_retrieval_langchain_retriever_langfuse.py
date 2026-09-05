"""SPEC §6.1, #28 — Langfuse observability: the retriever appears as a first-class
`retriever` observation (query in, `Document` objects out) with **no tracing code in
`PipelineRetriever` itself** — being a real `BaseRetriever` is what makes this work.
This test proves the wiring, it doesn't add any."""

from conftest import CreateCollection, raw_point, stub_embed
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from rag.retrieval.langchain_retriever import PipelineRetriever
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS


def test_invoke_with_a_langfuse_callback_produces_a_retriever_observation(
    qdrant: QdrantClient,
    create_collection: CreateCollection,
    langfuse_client: Langfuse,
    langfuse_span_exporter: InMemorySpanExporter,
) -> None:
    client = langfuse_client
    span_exporter = langfuse_span_exporter
    handler = CallbackHandler(public_key="test")

    create_collection(qdrant, FICHES_ALIAS)
    create_collection(qdrant, ARTICLES_ALIAS)
    qdrant.upsert(
        ARTICLES_ALIAS,
        points=[raw_point(1, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2", "text": "Le texte."})],
    )
    retriever = PipelineRetriever(
        client=qdrant, embed=stub_embed([1.0, 0.0, 0.0, 0.0]), lookup_keys=set()
    )

    question = "Quelle franchise pour un dégât des eaux ?"
    retriever.invoke(question, config={"callbacks": [handler]})
    client.flush()

    spans = span_exporter.get_finished_spans()
    [span] = spans
    assert span.attributes is not None

    assert span.attributes.get("langfuse.observation.type") == "retriever"
    assert span.attributes.get("langfuse.observation.input") == question
    output = span.attributes.get("langfuse.observation.output")
    assert isinstance(output, str) and "Le texte." in output
