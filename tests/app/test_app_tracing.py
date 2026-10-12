"""Prod tracing (SPEC §11.1, §14.7, #54) — against the real Langfuse SDK, exporting to memory
instead of the network, so what is pinned is the trace Langfuse would actually receive: one
`ask` trace per question, retrieval inside it as a `retriever` observation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

import rag.app.tracing as tracing_module
from rag.app.tracing import TRACE_NAME, TRACE_TAG, Tracing, make_tracing, untraced_turn
from rag.condensation.prompt import HistoryTurn
from rag.config import Settings
from rag.generation.citation import check_citations
from rag.generation.pipeline import GenerationResult
from rag.generation.schema import Reponse
from rag.ingest.upsert import EmbedFn
from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.pipeline import RetrievalResult
from rag.retrieval.short_circuit import ShortCircuitPath

ARTICLE = Candidate(
    id="point-L113-12",
    score=0.9,
    register=Register.ARTICLE,
    payload={"citation_id": "L113-12", "text": "..."},
    provenance=frozenset({Provenance.SEARCH}),
)
HISTORY = [HistoryTurn(role="user", content="franchise ?"), HistoryTurn(role="assistant", content="...")]


def _settings(*, tracing: bool, public_key: str = "pk-lf-test") -> Settings:
    return Settings.from_env(
        {"LANGFUSE_TRACING": "true" if tracing else "false", "LANGFUSE_PUBLIC_KEY": public_key}
    )


@pytest.fixture
def exporter(langfuse_span_exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    return langfuse_span_exporter


@pytest.fixture
def tracing(langfuse_client: Langfuse) -> Tracing:
    # The process-wide test client (tests/conftest.py): a second `Langfuse` would neither get
    # its own exporter nor leave `get_client()` usable for the other span tests.
    built = make_tracing(_settings(tracing=True, public_key="test"), langfuse=langfuse_client)
    assert built is not None
    return built


@pytest.fixture(autouse=True)
def fake_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    def retrieve(
        client: QdrantClient,
        embed: EmbedFn,
        raw_turn: str,
        lookup_keys: AbstractSet[str],
        *,
        arm: str,
    ) -> RetrievalResult:
        return RetrievalResult(short_circuit_path=ShortCircuitPath.NO_REFERENCE, contexts=[ARTICLE])

    monkeypatch.setattr(tracing_module, "retrieve", retrieve)


def _result() -> GenerationResult:
    envelope = Reponse(explanation="Oui.")
    return GenerationResult(
        envelope=envelope, citation_outcome=check_citations(envelope, [ARTICLE]), contexts=[ARTICLE]
    )


def _spans(tracing: Tracing, exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    tracing.langfuse.flush()
    return {span.name: span for span in exporter.get_finished_spans()}


def _attribute(span: ReadableSpan, name: str) -> Any:
    attributes: Mapping[str, Any] = span.attributes or {}
    return attributes[f"langfuse.{name}"]


def test_tracing_off_builds_nothing() -> None:
    assert make_tracing(_settings(tracing=False)) is None


def test_untraced_turn_records_nothing() -> None:
    with untraced_turn("question", HISTORY) as record:
        record(_result())


def test_a_turn_is_one_ask_trace_with_the_question_in_and_the_envelope_out(
    tracing: Tracing, exporter: InMemorySpanExporter
) -> None:
    with tracing.turn("et si je suis locataire ?", HISTORY) as record:
        record(_result())

    ask = _spans(tracing, exporter)[TRACE_NAME]
    assert ask.parent is None
    assert _attribute(ask, "trace.name") == TRACE_NAME
    assert _attribute(ask, "trace.tags") == (TRACE_TAG,)
    assert json.loads(_attribute(ask, "observation.input")) == {
        "question": "et si je suis locataire ?",
        "history": [{"role": "user", "content": "franchise ?"}, {"role": "assistant", "content": "..."}],
    }
    assert json.loads(_attribute(ask, "observation.output"))["explanation"] == "Oui."


def test_retrieval_is_a_retriever_observation_inside_the_turn(
    tracing: Tracing, exporter: InMemorySpanExporter
) -> None:
    with tracing.turn("et si je suis locataire ?", HISTORY):
        result = tracing.retrieve_fn(
            QdrantClient(":memory:"), lambda texts: [], "franchise si locataire ?", frozenset(), arm="rung1"
        )

    assert result.contexts == [ARTICLE]
    spans = _spans(tracing, exporter)
    retrieval = spans["retrieval"]
    assert retrieval.parent is not None
    assert retrieval.parent.span_id == spans[TRACE_NAME].context.span_id
    assert _attribute(retrieval, "observation.type") == "retriever"
    assert _attribute(retrieval, "observation.input") == "franchise si locataire ?"
    assert json.loads(_attribute(retrieval, "observation.output")) == [
        {"id": "point-L113-12", "register": "article", "citation_id": "L113-12", "score": 0.9}
    ]


def test_a_failed_turn_still_ends_its_trace(tracing: Tracing, exporter: InMemorySpanExporter) -> None:
    with pytest.raises(RuntimeError), tracing.turn("question", []):
        raise RuntimeError("OpenRouter down")

    assert TRACE_NAME in _spans(tracing, exporter)
