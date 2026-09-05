"""SPEC §6.1, #28 — the `BaseRetriever` wrapper around `rag.retrieval.pipeline.retrieve`.

No ranking, no filtering, no retries here — that all lives in `rag.retrieval.pipeline`.
This module's whole job is the `Candidate` -> `Document` conversion at the boundary
(SPEC §7.5: register/provenance are attached here, never stored, never computed twice).

**Rung 4 / #31.** SPEC §9.4/§11.1: the reranker is not a LangChain component, so
`rag.retrieval.rerank.rerank()` does not auto-trace the way `hybrid_leg`/`expand` do by
riding inside this class's own `retriever` observation span. `arm="rung4"` is therefore
special-cased in `_get_relevant_documents`: it calls `retrieve_rung4` directly (not the
generic `retrieve()` dispatcher) so a `rerank_fn` — wrapped in a Langfuse span by `_traced`
— can be injected the same way `retrieve_rung4` itself already allows tests to.

`_traced`'s span body was written by @Zameloth (LangChain/Langfuse portions are paired, not
agent-authored, per the #31 issue comment) — the field, the branch and the test
(`test_retrieval_langchain_retriever_rerank_span.py`) are plumbing, same posture as this
class's own `CallbackHandler` wiring was for #28.
"""

from __future__ import annotations

import functools
from collections.abc import Set as AbstractSet

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langfuse import get_client
from pydantic import ConfigDict
from qdrant_client import QdrantClient

from rag.ingest.upsert import EmbedFn
from rag.retrieval.candidates import Candidate
from rag.retrieval.pipeline import DEFAULT_RETRIEVAL_ARM, retrieve, retrieve_rung4
from rag.retrieval.rerank import DEFAULT_RERANKER_MODEL, RerankerBackend, RerankFn, rerank

__all__ = ["PipelineRetriever"]


class PipelineRetriever(BaseRetriever):
    # QdrantClient/EmbedFn aren't Pydantic-native types — BaseRetriever is a pydantic.BaseModel,
    # so without this, instantiating with client=... would raise a validation error.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Same four parameters as `retrieve()` itself (`pipeline.py`) — this class adds no
    # logic of its own, it's a pure adapter, so its fields mirror that function 1:1.
    client: QdrantClient
    embed: EmbedFn
    lookup_keys: AbstractSet[str]
    arm: str = DEFAULT_RETRIEVAL_ARM
    # SPEC §9.4/#31: the same injection seam `retrieve_rung4` itself takes (`RerankFn`
    # shaped) — resolved to the real `rerank()` when arm="rung4" and nothing is injected,
    # so tests never have to load the real cross-encoder checkpoint either.
    rerank_fn: RerankFn | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if self.arm == "rung4":
            inner = self.rerank_fn or functools.partial(
                rerank, model_id=DEFAULT_RERANKER_MODEL, backend=RerankerBackend.FP32
            )
            result = retrieve_rung4(
                self.client, self.embed, query, self.lookup_keys, rerank_fn=_traced(inner)
            )
        else:
            result = retrieve(self.client, self.embed, query, self.lookup_keys, arm=self.arm)
        return [_to_document(c) for c in result.contexts]


def _traced(inner: RerankFn) -> RerankFn:
    """Hand-wraps `inner` in a Langfuse span (SPEC §9.4, §11.1, #31): the reranker isn't a
    LangChain component, so its call doesn't ride inside this class's own auto-traced
    `retriever` observation the way `hybrid_leg`/`expand` do — without this, rung 4's own
    measurement is silently blind. `inner` is `RerankFn`-shaped so the real `rerank()` and a
    test's fake (`test_retrieval_langchain_retriever_rerank_span.py`) are wrapped
    identically; only what happens *inside* `wrapped` differs from a plain passthrough.

    The span records candidate ids on output, not full payloads — the reranked text itself
    already lives on the parent `retriever` span's own output, so repeating it here would
    just bloat the trace for no new information.
    """

    def wrapped(query: str, candidates: list[Candidate]) -> list[Candidate]:
        with get_client().start_as_current_observation(
            name="rerank",
            as_type="span",
            input=query,
            metadata={"candidate_count": len(candidates)},
        ) as span:
            result = inner(query, candidates)
            span.update(output=[c.id for c in result])
            return result

    return wrapped


def _to_document(candidate: Candidate) -> Document:
    return Document(
        id=candidate.id,
        page_content=str(candidate.payload["text"]),
        metadata={
            "register": candidate.register.value,
            "provenance": sorted(p.value for p in candidate.provenance),
        },
    )
