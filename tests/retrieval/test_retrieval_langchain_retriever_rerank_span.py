"""SPEC §9.4/§11.1, #31 — the hand-wrapped Langfuse span around the rung-4 reranker call.

`PipelineRetriever`'s own `retriever` observation (SPEC §6.1, #28,
`test_retrieval_langchain_retriever_langfuse.py`) already covers `_get_relevant_documents`
end to end. What this test proves is the *child* span `_traced` opens specifically around
the reranker step — the one measurement the parent span alone can't isolate, since the
reranker is not a LangChain component and so doesn't auto-trace the way `hybrid_leg`/
`expand` do (SPEC §9.4).

Never loads the real cross-encoder: `rerank_fn` is injected exactly as `retrieve_rung4`'s
own tests inject one (`test_retrieval_pipeline.py`'s `_reversing_rerank`).
"""

from __future__ import annotations

from conftest import CreateCollection, raw_point, stub_embed_hybrid
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from rag.retrieval.candidates import Candidate
from rag.retrieval.langchain_retriever import PipelineRetriever
from rag.retrieval.legs import ARTICLES_ALIAS, FICHES_ALIAS


def _fake_rerank(query: str, candidates: list[Candidate]) -> list[Candidate]:
    """Reverses the pool — same trick `_reversing_rerank` uses in
    `test_retrieval_pipeline.py` — so a passing test also proves the span wraps the call
    that actually reorders, not a no-op standing in for it."""
    return list(reversed(candidates))


def test_rung4_wraps_the_rerank_call_in_its_own_langfuse_span(
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
        points=[
            raw_point(1, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-2", "text": "Le texte."}),
            raw_point(2, [1.0, 0.0, 0.0, 0.0], {"citation_id": "L113-3", "text": "Un autre."}),
        ],
    )
    retriever = PipelineRetriever(
        client=qdrant,
        embed=stub_embed_hybrid([1.0, 0.0, 0.0, 0.0], SparseVector(indices=[1], values=[0.5])),
        lookup_keys=set(),
        arm="rung4",
        rerank_fn=_fake_rerank,
    )

    retriever.invoke("une question ouverte", config={"callbacks": [handler]})
    client.flush()

    spans = span_exporter.get_finished_spans()
    rerank_spans = [span for span in spans if span.name == "rerank"]
    assert len(rerank_spans) == 1, [span.name for span in spans]

    # A real child of the retriever span, not a second root — proves `_traced` opened it
    # from inside the already-current trace rather than starting a detached one.
    [retriever_span] = [span for span in spans if span.name != "rerank"]
    [rerank_span] = rerank_spans
    assert rerank_span.parent is not None
    assert rerank_span.parent.span_id == retriever_span.context.span_id
