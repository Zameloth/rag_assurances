"""SPEC §8.6, §8.7 — message assembly and the few-shot examples' independence from the
golden set."""

from __future__ import annotations

from pathlib import Path

import yaml

from rag.condensation.prompt import (
    MAX_HISTORY_TURN_CHARS,
    MAX_HISTORY_TURNS,
    SYSTEM_PROMPT,
    HistoryTurn,
    build_messages,
    trim_history,
)

GOLDEN_SET_PATH = Path(__file__).resolve().parents[2] / "eval" / "golden" / "golden-set.yaml"


def test_build_messages_with_no_history_is_system_then_raw_turn() -> None:
    messages = build_messages("Et si je suis locataire ?")

    assert messages == [
        ("system", SYSTEM_PROMPT),
        ("user", "Et si je suis locataire ?"),
    ]


def test_build_messages_carries_history_verbatim_before_the_raw_turn() -> None:
    history = [
        HistoryTurn(role="user", content="Est-ce que la garantie dégât des eaux couvre ça ?"),
        HistoryTurn(role="assistant", content="Oui, sous conditions."),
    ]

    messages = build_messages("Et si je suis locataire ?", history)

    assert messages == [
        ("system", SYSTEM_PROMPT),
        ("user", "Est-ce que la garantie dégât des eaux couvre ça ?"),
        ("assistant", "Oui, sous conditions."),
        ("user", "Et si je suis locataire ?"),
    ]


def test_few_shot_examples_are_not_drawn_from_the_golden_sets_multi_turn_items() -> None:
    """ADR-0008: seeding the prompt from the same 10 multi-turn items it is measured on
    would make the measurement partly one of memorisation — checked here against the real
    committed golden set, not a hand-picked excerpt of it."""
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        golden_set = yaml.safe_load(f)

    multi_turn_questions = {
        item["question"] for item in golden_set if "multi_turn" in item.get("tags", [])
    }
    multi_turn_history_content = {
        turn["content"]
        for item in golden_set
        if "multi_turn" in item.get("tags", [])
        for turn in item["history"]
    }

    assert multi_turn_questions, "expected at least one multi_turn golden item to compare against"
    for question in multi_turn_questions | multi_turn_history_content:
        assert question not in SYSTEM_PROMPT


# --- trim_history: SPEC §8.7's server-side enforcement ------------------------


def test_trim_history_passes_through_a_history_already_inside_both_bounds() -> None:
    history = [
        HistoryTurn(role="user", content="Une question ?"),
        HistoryTurn(role="assistant", content="Une réponse."),
    ]

    assert trim_history(history) == tuple(history)


def test_trim_history_keeps_only_the_last_max_history_turns() -> None:
    history = [HistoryTurn(role="user", content=f"question {i}") for i in range(20)]

    trimmed = trim_history(history)

    assert len(trimmed) == MAX_HISTORY_TURNS
    assert [t.content for t in trimmed] == [f"question {i}" for i in range(14, 20)]


def test_trim_history_clamps_each_turns_raw_size() -> None:
    runaway = "x" * (MAX_HISTORY_TURN_CHARS * 10)

    trimmed = trim_history([HistoryTurn(role="user", content=runaway)])

    assert len(trimmed[0].content) == MAX_HISTORY_TURN_CHARS


def test_trim_history_applies_both_caps_together() -> None:
    """Neither cap alone is enough (SPEC §8.7): a client posting many turns, each also
    oversized, must be bounded on both axes at once."""
    history = [
        HistoryTurn(role="user", content="x" * (MAX_HISTORY_TURN_CHARS * 10))
        for _ in range(50)
    ]

    trimmed = trim_history(history)

    assert len(trimmed) == MAX_HISTORY_TURNS
    assert all(len(t.content) == MAX_HISTORY_TURN_CHARS for t in trimmed)


def test_trim_history_preserves_order() -> None:
    history = [HistoryTurn(role="user", content=str(i)) for i in range(3)]

    assert [t.content for t in trim_history(history)] == ["0", "1", "2"]


def test_trim_history_of_empty_history_is_empty() -> None:
    assert trim_history([]) == ()
