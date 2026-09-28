"""`rag.pipeline.run_chain` — the one condense -> retrieve -> generate path the eval and
the app both run (CONTEXT.md: "the pipeline is a library", #50).

Retrieval is swapped out through `monkeypatch` rather than a real Qdrant: what these tests
pin is what `run_chain` hands each stage, not what retrieval returns.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet

import pytest
from qdrant_client import QdrantClient

import rag.pipeline as pipeline_module
from rag.condensation.prompt import MAX_HISTORY_TURN_CHARS, MAX_HISTORY_TURNS, HistoryTurn
from rag.condensation.prompt import Message as CondensationMessage
from rag.condensation.schema import CondenserOutput
from rag.generation.prompt import Message as GenerationMessage
from rag.generation.schema import Envelope, Reponse
from rag.ingest.upsert import EmbedFn
from rag.pipeline import run_chain
from rag.retrieval.pipeline import RetrievalResult
from rag.retrieval.short_circuit import ShortCircuitPath


def _long_history(turns: int) -> list[HistoryTurn]:
    return [
        HistoryTurn(role="user" if i % 2 == 0 else "assistant", content=f"tour-{i} " + "x" * 5000)
        for i in range(turns)
    ]


@pytest.fixture
def retrieved_queries(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    queries: list[str] = []

    def fake_retrieve(
        client: QdrantClient,
        embed: EmbedFn,
        raw_turn: str,
        lookup_keys: AbstractSet[str],
        *,
        arm: str,
    ) -> RetrievalResult:
        queries.append(raw_turn)
        return RetrievalResult(short_circuit_path=ShortCircuitPath.NO_REFERENCE, contexts=[])

    monkeypatch.setattr(pipeline_module, "retrieve", fake_retrieve)
    return queries


def test_generation_sees_the_same_trimmed_history_the_condenser_does(
    retrieved_queries: list[str],
) -> None:
    """SPEC §8.7: history is untrusted client input — a 200-turn, 5k-char-per-turn history
    must never reach the generation prompt either, not just the condenser's."""
    seen_by_generation: list[list[GenerationMessage]] = []

    def condense_fn(messages: list[CondensationMessage]) -> CondenserOutput:
        return CondenserOutput(requete="question autonome ?")

    def generate_fn(messages: list[GenerationMessage]) -> Envelope:
        seen_by_generation.append(messages)
        return Reponse(explanation="...")

    run_chain(
        "et si je suis locataire ?",
        _long_history(200),
        client=QdrantClient(":memory:"),
        embed=lambda texts: [],
        lookup_keys=frozenset(),
        condense_fn=condense_fn,
        generate_fn=generate_fn,
    )

    [messages] = seen_by_generation
    history_messages = [content for role, content in messages[1:-1]]
    assert len(history_messages) == MAX_HISTORY_TURNS
    assert history_messages[0].startswith("tour-194 ")
    assert all(len(content) <= MAX_HISTORY_TURN_CHARS for content in history_messages)
    assert retrieved_queries == ["question autonome ?"]


def test_generation_gets_the_raw_turn_not_the_condensed_query(
    retrieved_queries: list[str],
) -> None:
    """ADR-0008: the condensed query only ever reaches retrieval."""
    seen_by_generation: list[list[GenerationMessage]] = []

    def generate_fn(messages: list[GenerationMessage]) -> Envelope:
        seen_by_generation.append(messages)
        return Reponse(explanation="...")

    run_chain(
        "et si je suis locataire ?",
        [HistoryTurn(role="user", content="franchise ?"), HistoryTurn(role="assistant", content="...")],
        client=QdrantClient(":memory:"),
        embed=lambda texts: [],
        lookup_keys=frozenset(),
        condense_fn=lambda messages: CondenserOutput(requete="franchise si locataire ?"),
        generate_fn=generate_fn,
    )

    [messages] = seen_by_generation
    assert messages[-1] == ("user", "et si je suis locataire ?")
    assert retrieved_queries == ["franchise si locataire ?"]
