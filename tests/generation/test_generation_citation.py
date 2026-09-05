"""SPEC §10.5, ADR-0009, #42 — `cited ⊆ retrieved_context`, not `⊆ corpus`, never
auto-repaired; `repair_for_display` is the demo-path-only exception."""

from __future__ import annotations

from candidates import make_article, no_article_marker

from rag.generation.citation import (
    check_citations,
    repair_for_display,
    retrieved_citation_ids,
)
from rag.generation.schema import FondementJuridique, Motif, Refus, Reponse
from rag.retrieval.quota import NO_ARTICLE_MARKER_TEXT


def test_retrieved_citation_ids_excludes_the_no_article_marker() -> None:
    contexts = [make_article("L113-15-2"), no_article_marker()]

    assert retrieved_citation_ids(contexts) == frozenset({"L113-15-2"})


def test_citation_within_retrieved_context_is_valid() -> None:
    envelope = Reponse(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L113-15-2", gloss="résiliation")],
    )
    contexts = [make_article("L113-15-2")]

    outcome = check_citations(envelope, contexts)

    assert outcome.valid
    assert outcome.fabricated_ids == frozenset()


def test_real_but_unretrieved_article_is_fabrication_not_corpus_membership() -> None:
    """SPEC §10.5: `⊆ retrieved_context`, not `⊆ corpus` — a real article the pipeline
    never retrieved this turn still fails the guardrail."""
    envelope = Reponse(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L121-1", gloss="indemnisation")],
    )
    contexts = [make_article("L113-15-2")]  # L121-1 exists in the corpus, just not retrieved here

    outcome = check_citations(envelope, contexts)

    assert not outcome.valid
    assert outcome.fabricated_ids == frozenset({"L121-1"})


def test_refus_citations_are_checked_too() -> None:
    envelope = Refus(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L211-1", gloss="obligation")],
        motif=Motif.RECOMMANDATION_PRODUIT,
    )

    outcome = check_citations(envelope, [make_article("L211-1")])

    assert outcome.valid


def test_repair_for_display_is_a_noop_when_valid() -> None:
    envelope = Reponse(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L113-15-2", gloss="résiliation")],
    )
    outcome = check_citations(envelope, [make_article("L113-15-2")])

    assert repair_for_display(envelope, outcome) == envelope


def test_repair_for_display_drops_the_fabricated_citation_only() -> None:
    envelope = Reponse(
        explanation="...",
        fondement_juridique=[
            FondementJuridique(article_id="L113-15-2", gloss="résiliation"),
            FondementJuridique(article_id="L121-1", gloss="fabriquée"),
        ],
    )
    outcome = check_citations(envelope, [make_article("L113-15-2")])

    repaired = repair_for_display(envelope, outcome)

    assert isinstance(repaired, Reponse)
    assert [f.article_id for f in repaired.fondement_juridique] == ["L113-15-2"]
    assert repaired.aucun_fondement is None


def test_repair_for_display_surfaces_the_no_article_marker_when_nothing_survives() -> None:
    envelope = Reponse(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L121-1", gloss="fabriquée")],
    )
    outcome = check_citations(envelope, [make_article("L113-15-2")])

    repaired = repair_for_display(envelope, outcome)

    assert isinstance(repaired, Reponse)
    assert repaired.fondement_juridique == []
    assert repaired.aucun_fondement == NO_ARTICLE_MARKER_TEXT


def test_repair_for_display_on_refus_drops_the_citation_without_a_marker() -> None:
    """`Refus` has no `aucun_fondement` field — dropping a bad citation there is not the
    no-article state, it's still a refusal."""
    envelope = Refus(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L121-1", gloss="fabriquée")],
        motif=Motif.HORS_CORPUS,
    )
    outcome = check_citations(envelope, [])

    repaired = repair_for_display(envelope, outcome)

    assert isinstance(repaired, Refus)
    assert repaired.fondement_juridique == []
    assert repaired.motif == envelope.motif


def test_check_citations_never_mutates_the_envelope() -> None:
    envelope = Reponse(
        explanation="...",
        fondement_juridique=[FondementJuridique(article_id="L121-1", gloss="fabriquée")],
    )
    before = envelope.model_copy(deep=True)

    check_citations(envelope, [])

    assert envelope == before
