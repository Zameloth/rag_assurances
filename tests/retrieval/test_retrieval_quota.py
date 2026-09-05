"""SPEC §9.5, ADR-0019, #32 — the rung-5 arm's register quota, pure Python against
synthetic candidates. Never touches Qdrant: the contract under test is `assemble_quota`'s
own slot-filling logic, not any store's ranking (SPEC §6.3)."""

from __future__ import annotations

from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.quota import (
    ARTICLE_QUOTA,
    ARTICLE_RELEVANCE_FLOOR,
    EXPANSION_PREFERENCE_MARGIN,
    FICHE_QUOTA,
    NO_ARTICLE_MARKER_ID,
    NO_ARTICLE_MARKER_TEXT,
    assemble_quota,
)


def _fiche(id_: str, score: float) -> Candidate:
    return Candidate(
        id=id_,
        score=score,
        register=Register.FICHE,
        payload={"text": id_},
        provenance=frozenset({Provenance.SEARCH}),
    )


def _article(
    id_: str,
    score: float,
    *,
    expansion: bool = False,
    provenance: frozenset[Provenance] | None = None,
) -> Candidate:
    if provenance is None:
        provenance = (
            frozenset({Provenance.EXPANSION}) if expansion else frozenset({Provenance.SEARCH})
        )
    return Candidate(
        id=id_, score=score, register=Register.ARTICLE, payload={"text": id_}, provenance=provenance
    )


def test_fiche_slots_are_a_plain_top_n_cap_no_floor_no_preference() -> None:
    fiches = [_fiche(str(i), float(i)) for i in range(6)]

    result = assemble_quota(fiches)

    fiche_ids = [c.id for c in result.contexts if c.register is Register.FICHE]
    assert fiche_ids == ["5", "4", "3", "2"]
    assert len(fiche_ids) == FICHE_QUOTA


def test_expansion_sourced_wins_a_tiebreak_within_the_margin() -> None:
    """SPEC §9.5: "article slots prefer expansion-sourced over search-sourced" — read as a
    tiebreak (ADR-0019), not a hard partition. A search-sourced article scoring within
    `expansion_preference_margin` of an expansion-sourced one loses the tie."""
    candidates = [
        _article("search-close", 0.6, expansion=False),
        _article("expansion-close", 0.55, expansion=True),
    ]
    assert EXPANSION_PREFERENCE_MARGIN > 0.6 - 0.55

    result = assemble_quota(candidates)

    assert [c.id for c in result.contexts] == ["expansion-close", "search-close"]
    assert result.floor_met is True


def test_search_sourced_wins_outright_when_its_lead_exceeds_the_margin() -> None:
    """A search-sourced article with a lead bigger than the margin is not overtaken —
    the preference only breaks ties among comparably-scored candidates."""
    candidates = [
        _article("search-high", 0.9, expansion=False),
        _article("expansion-low", 0.6, expansion=True),
    ]
    assert EXPANSION_PREFERENCE_MARGIN < 0.9 - 0.6

    result = assemble_quota(candidates)

    assert [c.id for c in result.contexts] == ["search-high", "expansion-low"]
    assert result.floor_met is True


def test_expansion_preference_margin_is_overridable_by_the_caller() -> None:
    """Widening the margin turns a would-be outright win into a tiebreak the
    expansion-sourced candidate wins — this ticket's own acceptance criterion that the
    preference stays reachable through config."""
    candidates = [
        _article("search-high", 0.9, expansion=False),
        _article("expansion-low", 0.6, expansion=True),
    ]

    result = assemble_quota(candidates, expansion_preference_margin=0.5)

    assert [c.id for c in result.contexts] == ["expansion-low", "search-high"]


def test_relevance_floor_rejects_rather_than_pads() -> None:
    """This ticket's own acceptance criterion: fewer than 4 articles is a valid outcome —
    a below-floor candidate is dropped, never used to fill out the remaining slots."""
    candidates = [
        _article("above", 0.9),
        _article("below", 0.1),
    ]

    result = assemble_quota(candidates)

    assert [c.id for c in result.contexts] == ["above"]
    assert result.floor_met is True


def test_default_relevance_floor_is_the_sigmoid_midpoint() -> None:
    assert ARTICLE_RELEVANCE_FLOOR == 0.5


def test_no_article_clears_the_floor_inserts_the_marker() -> None:
    candidates = [_fiche("f1", 0.9), _article("weak", 0.2)]

    result = assemble_quota(candidates)

    assert result.floor_met is False
    [fiche_ctx, article_ctx] = result.contexts
    assert fiche_ctx.id == "f1"
    assert article_ctx.id == NO_ARTICLE_MARKER_ID
    assert article_ctx.register is Register.ARTICLE
    assert article_ctx.payload["text"] == NO_ARTICLE_MARKER_TEXT
    assert article_ctx.provenance == frozenset()


def test_no_articles_at_all_also_inserts_the_marker() -> None:
    result = assemble_quota([_fiche("f1", 0.9)])

    assert result.floor_met is False
    assert [c.id for c in result.contexts] == ["f1", NO_ARTICLE_MARKER_ID]


def test_article_quota_caps_even_when_all_clear_the_floor() -> None:
    candidates = [_article(str(i), 0.9 - i * 0.01) for i in range(6)]

    result = assemble_quota(candidates)

    assert len(result.contexts) == ARTICLE_QUOTA
    assert [c.id for c in result.contexts] == ["0", "1", "2", "3"]


def test_quota_depths_and_floor_are_overridable_by_the_caller() -> None:
    """Quota depth and the floor value are config, not code (this ticket's own acceptance
    criterion) — a caller overrides the defaults without editing `quota.py`."""
    candidates = [_fiche(str(i), float(i)) for i in range(3)] + [
        _article("a1", 0.4),
        _article("a2", 0.3),
    ]

    result = assemble_quota(candidates, fiche_quota=2, article_quota=1, relevance_floor=0.35)

    fiche_ids = [c.id for c in result.contexts if c.register is Register.FICHE]
    article_ids = [c.id for c in result.contexts if c.register is Register.ARTICLE]
    assert fiche_ids == ["2", "1"]
    assert article_ids == ["a1"]


def test_input_order_does_not_matter_assemble_quota_sorts_itself() -> None:
    candidates = [_fiche("low", 0.1), _fiche("high", 0.9)]

    result = assemble_quota(candidates, fiche_quota=1)

    fiche_ids = [c.id for c in result.contexts if c.register is Register.FICHE]
    assert fiche_ids == ["high"]
