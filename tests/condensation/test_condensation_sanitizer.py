"""SPEC §8.3's sanitizer table and §8.4's reference monotonicity."""

from __future__ import annotations

import pytest

from rag.condensation.sanitizer import MAX_CONDENSED_CHARS, sanitize_condensed


def test_accepts_a_clean_single_line_rewrite() -> None:
    result = sanitize_condensed(
        "Et si je suis locataire ?",
        "Est-ce que la garantie dégât des eaux couvre ça si je suis locataire ?",
    )

    assert result == "Est-ce que la garantie dégât des eaux couvre ça si je suis locataire ?"


def test_trims_surrounding_whitespace() -> None:
    result = sanitize_condensed("question", "  Une question ?  ")

    assert result == "Une question ?"


@pytest.mark.parametrize("candidate", ["", "   ", "\n"])
def test_rejects_empty_or_whitespace_only(candidate: str) -> None:
    assert sanitize_condensed("question", candidate) is None


def test_rejects_multi_line_output() -> None:
    assert sanitize_condensed("question", "Voici la question :\nUne question ?") is None


def test_a_single_trailing_newline_is_not_multi_line() -> None:
    """Stripped away before the multi-line check — only an *embedded* newline counts."""
    assert sanitize_condensed("question", "Une question ?\n") == "Une question ?"


def test_rejects_runaway_output_over_the_length_cap() -> None:
    candidate = "Une question " + "très " * 100 + "longue ?"
    assert len(candidate) > MAX_CONDENSED_CHARS
    assert sanitize_condensed("question", candidate) is None


def test_accepts_output_right_at_the_length_cap() -> None:
    candidate = "a" * MAX_CONDENSED_CHARS
    assert sanitize_condensed("question", candidate) == candidate


def test_rejects_a_manufactured_article_reference_not_in_the_raw_turn() -> None:
    """SPEC §8.4: `refs(condensed) ⊆ refs(raw_turn)` — a hallucinated reference must fall
    back even though it is a perfectly well-formed single-line string."""
    result = sanitize_condensed(
        "Et si je résilie maintenant ?",
        "Que dit L121-1 si je résilie maintenant ?",
    )

    assert result is None


def test_accepts_a_reference_that_was_already_in_the_raw_turn() -> None:
    result = sanitize_condensed(
        "Et pour L113-15-2, la garantie change si je suis locataire ?",
        "La garantie change-t-elle pour L113-15-2 si je suis locataire ?",
    )

    assert result == "La garantie change-t-elle pour L113-15-2 si je suis locataire ?"


def test_reference_comparison_is_normalized_not_a_literal_string_match() -> None:
    """`L. 113-15` and `L113-15` normalize to the same `lookup_key` — the same normalizer
    the short-circuit's field validator uses (`rag.ingest.lookup_key`)."""
    result = sanitize_condensed(
        "Et pour L. 113-15, ça change si je suis locataire ?",
        "La garantie change-t-elle pour L113-15 si je suis locataire ?",
    )

    assert result == "La garantie change-t-elle pour L113-15 si je suis locataire ?"


def test_rejects_when_raw_turn_has_no_reference_at_all() -> None:
    """The empty-set edge case: `refs(raw_turn) == set()` forces `refs(condensed)` to be
    empty too — any reference at all in the rewrite must fall back."""
    result = sanitize_condensed(
        "Et si je suis locataire ?",
        "Que dit L113-15 si je suis locataire ?",
    )

    assert result is None
