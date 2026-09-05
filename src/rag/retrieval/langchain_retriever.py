"""SPEC §6.1, #28 — the `BaseRetriever` wrapper around `rag.retrieval.pipeline.retrieve`.

No ranking, no filtering, no retries here — that all lives in `rag.retrieval.pipeline`.
This module's whole job is the `Candidate` -> `Document` conversion at the boundary
(SPEC §7.5: register/provenance are attached here, never stored, never computed twice).
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client import QdrantClient

from rag.ingest.upsert import EmbedFn
from rag.retrieval.candidates import Candidate
from rag.retrieval.pipeline import DEFAULT_RETRIEVAL_ARM, retrieve

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

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        result = retrieve(self.client, self.embed, query, self.lookup_keys, arm=self.arm)
        return [_to_document(c) for c in result.contexts]


def _to_document(candidate: Candidate) -> Document:
    return Document(
        id=candidate.id,
        page_content=str(candidate.payload["text"]),
        metadata={
            "register": candidate.register.value,
            "provenance": sorted(p.value for p in candidate.provenance),
        },
    )
