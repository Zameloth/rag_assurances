"""SPEC §10.6, ADR-0009, #42 — context grouped by register with `citation_id` as the
label, the article breadcrumb minus its first segment, named exclusions, and history
stripped of citations by construction."""

from __future__ import annotations

from candidates import make_article, make_fiche, no_article_marker

from rag.generation.prompt import (
    ARTICLE_HEADING,
    FICHE_HEADING,
    SYSTEM_PROMPT,
    HistoryTurn,
    build_context_section,
    build_messages,
    strip_breadcrumb_root,
)
from rag.retrieval.quota import NO_ARTICLE_MARKER_TEXT


def test_strip_breadcrumb_root_drops_only_the_first_segment() -> None:
    breadcrumb = (
        "Partie législative > Livre Ier : Le contrat > "
        "Titre Ier : Règles communes aux assurances de dommages > "
        "Chapitre IV : Résiliation"
    )

    assert strip_breadcrumb_root(breadcrumb) == (
        "Livre Ier : Le contrat > "
        "Titre Ier : Règles communes aux assurances de dommages > "
        "Chapitre IV : Résiliation"
    )


def test_strip_breadcrumb_root_handles_missing_breadcrumb() -> None:
    assert strip_breadcrumb_root(None) == ""
    assert strip_breadcrumb_root("") == ""


def test_context_section_groups_by_register_under_named_headings() -> None:
    fiche = make_fiche(
        title="Modification du contrat d'assurance habitation",
        cas_label="Si vous êtes locataire",
        text="Vous devez prévenir l'assureur dans les quinze jours.",
    )
    article = make_article(
        "L113-15-2",
        full_sections_titre="Partie législative > Livre Ier : Le contrat > Chapitre IV : Résiliation",
        text="Le contrat peut être résilié par l'assuré...",
    )

    section = build_context_section([fiche, article])

    assert section.index(FICHE_HEADING) < section.index(ARTICLE_HEADING)
    assert "Modification du contrat d'assurance habitation" in section
    assert "Si vous êtes locataire" in section
    assert "L113-15-2" in section
    assert "Livre Ier : Le contrat > Chapitre IV : Résiliation" in section
    # first breadcrumb segment is dropped
    assert "Partie législative >" not in section


def test_citation_id_is_the_label_no_bracket_handles() -> None:
    article = make_article("L113-15-2")

    section = build_context_section([article])

    assert "[A1]" not in section
    assert "[F1]" not in section
    assert "L113-15-2" in section


def test_excluded_fields_never_appear_in_the_prompt() -> None:
    fiche = make_fiche(
        title="Titre",
        text="texte",
        fiche_id="F2123",
        sp_url="https://service-public.fr/fiche/F2123",
        date_modified="2026-01-01",
    )
    article = make_article(
        "L113-15-2",
        legiarti_version_id="LEGIARTI000006791829",
        date_debut="2020-01-01",
    )

    section = build_context_section([fiche, article])

    assert "F2123" not in section
    assert "https://" not in section
    assert "2026-01-01" not in section
    assert "LEGIARTI000006791829" not in section
    assert "2020-01-01" not in section


def test_no_article_marker_renders_without_a_citation_label() -> None:
    section = build_context_section([no_article_marker()])

    assert ARTICLE_HEADING in section
    assert NO_ARTICLE_MARKER_TEXT in section


def test_empty_register_omits_its_heading() -> None:
    section = build_context_section([make_article("L113-15-2")])

    assert FICHE_HEADING not in section
    assert ARTICLE_HEADING in section


def test_build_messages_leads_with_the_system_prompt() -> None:
    messages = build_messages("une question", [])

    assert messages[0] == ("system", SYSTEM_PROMPT)


def test_system_prompt_pins_french_explicitly() -> None:
    assert "français" in SYSTEM_PROMPT.lower()


def test_build_messages_carries_history_verbatim_between_system_and_current_turn() -> None:
    history = [
        HistoryTurn(role="user", content="comment marche la franchise ?"),
        HistoryTurn(role="assistant", content="La franchise est le montant restant à votre charge."),
    ]

    messages = build_messages("et si je suis locataire ?", [], history=history)

    assert messages[1] == ("user", "comment marche la franchise ?")
    assert messages[2] == (
        "assistant",
        "La franchise est le montant restant à votre charge.",
    )
    assert messages[-1][0] == "user"
    assert "et si je suis locataire ?" in messages[-1][1]


def test_history_turn_has_no_field_to_carry_a_prior_citation_list() -> None:
    """SPEC §10.6's history stripping is load-bearing: `HistoryTurn` only ever has
    `role`/`content`, so a prior `fondement_juridique` list has nothing to ride in on."""
    assert set(HistoryTurn.__dataclass_fields__) == {"role", "content"}


def test_build_messages_without_contexts_sends_the_raw_turn_untouched() -> None:
    messages = build_messages("bonjour", [])

    assert messages[-1] == ("user", "bonjour")
