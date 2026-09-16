"""SPEC §8.1, §8.8, ADR-0008 — `condense()` wires the two skips, the injected `condense_fn`
seam and the sanitizer into the six-path `condense_status`."""

from __future__ import annotations

from rag.condensation.pipeline import CondenseFn, CondenseStatus, condense
from rag.condensation.prompt import MAX_HISTORY_TURNS, HistoryTurn, Message
from rag.condensation.schema import CondenserOutput

_HISTORY = [
    HistoryTurn(role="user", content="Est-ce que la garantie dégât des eaux couvre ça ?"),
    HistoryTurn(role="assistant", content="Oui, sous conditions."),
]


def _fake_condense_fn(requete: str) -> CondenseFn:
    def fn(messages: list[Message]) -> CondenserOutput:
        return CondenserOutput(requete=requete)

    return fn


def _raising_condense_fn(exc: Exception) -> CondenseFn:
    def fn(messages: list[Message]) -> CondenserOutput:
        raise exc

    return fn


# --- history == [] ------------------------------------------------------------


def test_skips_condensation_when_history_is_empty() -> None:
    result = condense("Et si je suis locataire ?", [], frozenset(), _fake_condense_fn("ignored"))

    assert result.query == "Et si je suis locataire ?"
    assert result.condensed_query is None
    assert result.condense_status is CondenseStatus.SKIPPED_NO_HISTORY


# --- short-circuit path 1 (RESOLVED) -------------------------------------------


def test_skips_condensation_when_the_short_circuit_resolves() -> None:
    raw_turn = "Que dit L113-15 sur la résiliation ?"

    result = condense(raw_turn, _HISTORY, frozenset({"L113-15"}), _fake_condense_fn("ignored"))

    assert result.query == raw_turn
    assert result.condensed_query is None
    assert result.condense_status is CondenseStatus.SKIPPED_SHORT_CIRCUIT


def test_runs_condensation_when_short_circuit_membership_fails() -> None:
    """SPEC §8.1 only names path 1 (RESOLVED) as a skip — a reference present but not a
    real `lookup_key` (path 2, MEMBERSHIP_FAILED) still condenses."""
    raw_turn = "Que dit L999-99 sur la résiliation ?"

    result = condense(raw_turn, _HISTORY, frozenset({"L113-15"}), _fake_condense_fn(raw_turn))

    assert result.condense_status is not CondenseStatus.SKIPPED_SHORT_CIRCUIT


# --- the condenser call itself -------------------------------------------------


def test_passthrough_when_the_condenser_returns_the_raw_turn_verbatim() -> None:
    raw_turn = "Est-ce que la garantie bris de glace est incluse dans une assurance tous risques ?"

    result = condense(raw_turn, _HISTORY, frozenset(), _fake_condense_fn(raw_turn))

    assert result.query == raw_turn
    assert result.condensed_query == raw_turn
    assert result.condense_status is CondenseStatus.PASSTHROUGH


def test_rewritten_when_the_condenser_produces_a_clean_different_rewrite() -> None:
    rewrite = "Est-ce que la garantie dégât des eaux couvre ça si je suis locataire ?"

    result = condense(
        "Et si je suis locataire ?", _HISTORY, frozenset(), _fake_condense_fn(rewrite)
    )

    assert result.query == rewrite
    assert result.condensed_query == rewrite
    assert result.condense_status is CondenseStatus.REWRITTEN


def test_fallback_sanitizer_when_the_rewrite_is_rejected_but_still_records_it() -> None:
    bad_rewrite = "Que dit L121-1 si je suis locataire ?"  # manufactured reference

    result = condense(
        "Et si je suis locataire ?", _HISTORY, frozenset(), _fake_condense_fn(bad_rewrite)
    )

    assert result.query == "Et si je suis locataire ?"
    assert result.condensed_query == bad_rewrite
    assert result.condense_status is CondenseStatus.FALLBACK_SANITIZER


def test_fallback_error_when_the_condense_fn_raises() -> None:
    result = condense(
        "Et si je suis locataire ?",
        _HISTORY,
        frozenset(),
        _raising_condense_fn(TimeoutError("pinned endpoint down")),
    )

    assert result.query == "Et si je suis locataire ?"
    assert result.condensed_query is None
    assert result.condense_status is CondenseStatus.FALLBACK_ERROR


def test_history_is_trimmed_before_it_reaches_the_condenser_call() -> None:
    """SPEC §8.7's server-side enforcement happens inside `condense()` itself, not only in
    a caller a future app ticket might add — a client-supplied history longer than
    `MAX_HISTORY_TURNS` must never reach `build_messages`/`condense_fn` untrimmed."""
    oversized_history = [
        HistoryTurn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
        for i in range(30)
    ]
    captured: list[list[Message]] = []

    def capturing_fn(messages: list[Message]) -> CondenserOutput:
        captured.append(messages)
        return CondenserOutput(requete="Une question ?")

    condense("Et si je suis locataire ?", oversized_history, frozenset(), capturing_fn)

    messages = captured[0]
    # system + trimmed history + the raw turn itself
    assert len(messages) == MAX_HISTORY_TURNS + 2
    assert "turn 0" not in [content for (_, content) in messages]


def test_build_messages_receives_the_raw_turn_and_the_history() -> None:
    captured: list[list[Message]] = []

    def capturing_fn(messages: list[Message]) -> CondenserOutput:
        captured.append(messages)
        return CondenserOutput(requete="Une question ?")

    condense("Et si je suis locataire ?", _HISTORY, frozenset(), capturing_fn)

    assert len(captured) == 1
    messages = captured[0]
    assert messages[-1] == ("user", "Et si je suis locataire ?")
    assert ("user", _HISTORY[0].content) in messages
    assert ("assistant", _HISTORY[1].content) in messages
