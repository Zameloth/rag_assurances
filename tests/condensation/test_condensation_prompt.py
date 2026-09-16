"""SPEC §8.6, §8.7 — message assembly and the few-shot examples' independence from the
golden set."""

from __future__ import annotations

from pathlib import Path

import yaml

from rag.condensation.prompt import SYSTEM_PROMPT, HistoryTurn, build_messages

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
