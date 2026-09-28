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
from rag.condensation.pipeline import CondenseFn
from rag.condensation.prompt import MAX_HISTORY_TURN_CHARS, MAX_HISTORY_TURNS, HistoryTurn
from rag.condensation.prompt import Message as CondensationMessage
from rag.condensation.schema import CondenserOutput
from rag.generation.prompt import Message as GenerationMessage
from rag.generation.schema import Envelope, Reponse
from rag.ingest.upsert import EmbedFn
from rag.pipeline import Stage, run_chain
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


# --- stage events (SPEC §13.2, #51) --------------------------------------------

_HISTORY = [HistoryTurn(role="user", content="franchise ?"), HistoryTurn(role="assistant", content="...")]


def _run_recording_stages(
    raw_turn: str,
    history: list[HistoryTurn],
    *,
    lookup_keys: frozenset[str] = frozenset(),
    condense_fn: CondenseFn | None = None,
) -> list[Stage]:
    stages: list[Stage] = []

    def default_condense_fn(messages: list[CondensationMessage]) -> CondenserOutput:
        return CondenserOutput(requete=raw_turn)

    run_chain(
        raw_turn,
        history,
        client=QdrantClient(":memory:"),
        embed=lambda texts: [],
        lookup_keys=lookup_keys,
        condense_fn=condense_fn or default_condense_fn,
        generate_fn=lambda messages: Reponse(explanation="..."),
        on_stage=stages.append,
    )
    return stages


@pytest.mark.usefixtures("retrieved_queries")
class TestStageEvents:
    def test_a_follow_up_runs_all_three_chain_stages_in_order(self) -> None:
        stages = _run_recording_stages("et si je suis locataire ?", _HISTORY)
        assert stages == [Stage.CONDENSATION, Stage.RECHERCHE, Stage.GENERATION]

    def test_a_first_turn_does_not_claim_to_have_condensed(self) -> None:
        """Condensation fires only when history exists (ADR-0008)."""
        assert _run_recording_stages("puis-je résilier ?", []) == [Stage.RECHERCHE, Stage.GENERATION]

    def test_a_short_circuited_follow_up_does_not_claim_to_have_condensed(self) -> None:
        stages = _run_recording_stages(
            "Que dit L113-15 sur la résiliation ?", _HISTORY, lookup_keys=frozenset({"L113-15"})
        )
        assert stages == [Stage.RECHERCHE, Stage.GENERATION]

    def test_a_failed_condenser_call_still_ran(self) -> None:
        """`FALLBACK_ERROR` is a call that was made — the wait was real."""

        def failing(messages: list[CondensationMessage]) -> CondenserOutput:
            raise TimeoutError

        stages = _run_recording_stages("et si je suis locataire ?", _HISTORY, condense_fn=failing)
        assert stages[0] is Stage.CONDENSATION

    def test_each_stage_is_announced_as_it_starts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Emitted by the pipeline as it advances, not fabricated around it: each stage's
        event has already fired when its own work begins, and the next one has not."""
        stages: list[Stage] = []
        seen_at: dict[str, list[Stage]] = {}

        def condense_fn(messages: list[CondensationMessage]) -> CondenserOutput:
            seen_at["condense"] = list(stages)
            return CondenserOutput(requete="franchise si locataire ?")

        def fake_retrieve(*args: object, **kwargs: object) -> RetrievalResult:
            seen_at["retrieve"] = list(stages)
            return RetrievalResult(short_circuit_path=ShortCircuitPath.NO_REFERENCE, contexts=[])

        def generate_fn(messages: list[GenerationMessage]) -> Envelope:
            seen_at["generate"] = list(stages)
            return Reponse(explanation="...")

        monkeypatch.setattr(pipeline_module, "retrieve", fake_retrieve)
        run_chain(
            "et si je suis locataire ?",
            _HISTORY,
            client=QdrantClient(":memory:"),
            embed=lambda texts: [],
            lookup_keys=frozenset(),
            condense_fn=condense_fn,
            generate_fn=generate_fn,
            on_stage=stages.append,
        )
        assert seen_at == {
            "condense": [Stage.CONDENSATION],
            "retrieve": [Stage.CONDENSATION, Stage.RECHERCHE],
            "generate": [Stage.CONDENSATION, Stage.RECHERCHE, Stage.GENERATION],
        }
