"""Ingest assertions 1-4 (SPEC §7.4) — the gate the fetch script runs before it writes
`articles.jsonl`, and the same gate the committed corpus is checked against in
test_articles_corpus.py.
"""

from collections.abc import Mapping
from typing import Any

import pytest

from rag.ingest.assertions import CorpusAssertionError, run_article_assertions, run_fiche_assertions


def article(cid: str, citation_id: str | None = "L113-3", etat: str = "VIGUEUR") -> Mapping[str, Any]:
    return {"cid": cid, "citation_id": citation_id, "etat": etat}


def article_with_section(cid: str, section_parent_id: str | None) -> Mapping[str, Any]:
    return {"cid": cid, "sectionParentId": section_parent_id}


def fiche(fiche_id: str, section_ids: list[str]) -> Mapping[str, Any]:
    return {"fiche_id": fiche_id, "section_ids": section_ids}


def test_passes_on_a_clean_corpus() -> None:
    run_article_assertions(
        [
            article("cid-1", "L113-3"),
            article("cid-2", "L113-4"),
            # A prose annexe label — non-null citation_id, null lookup_key, no collision.
            article("cid-3", "Annexe à l'article A121-1"),
        ]
    )


def test_assertion_1_rejects_a_repeated_cid() -> None:
    with pytest.raises(CorpusAssertionError, match="assertion 1"):
        run_article_assertions([article("cid-1", "L113-3"), article("cid-1", "L113-4")])


def test_assertion_2_rejects_a_non_vigueur_row() -> None:
    with pytest.raises(CorpusAssertionError, match="assertion 2"):
        run_article_assertions([article("cid-1", "L113-3", etat="ABROGE_DIFF")])


class TestAssertion3LookupKeysValid:
    def test_rejects_two_citation_ids_normalizing_to_the_same_key(self) -> None:
        """`R*113-4` and `R.113-4` both normalize to `R113-4` — a real collision."""
        with pytest.raises(CorpusAssertionError, match="assertion 3"):
            run_article_assertions(
                [article("cid-1", "R*113-4"), article("cid-2", "R.113-4")]
            )

    def test_does_not_count_two_null_lookup_keys_as_a_collision(self) -> None:
        """Two annexes both normalize to `None` — `None` is not a duplicate key."""
        run_article_assertions(
            [
                article("cid-1", "Annexe à l'article A121-1"),
                article("cid-2", "Annexe à l'article A160-1"),
            ]
        )


class TestAssertion4CitationIdsPresent:
    def test_rejects_a_null_citation_id(self) -> None:
        with pytest.raises(CorpusAssertionError, match="assertion 4"):
            run_article_assertions([article("cid-1", None)])

    def test_rejects_an_empty_citation_id(self) -> None:
        with pytest.raises(CorpusAssertionError, match="assertion 4"):
            run_article_assertions([article("cid-1", "")])


class TestAssertion5FicheSectionsResolve:
    def test_passes_when_at_least_one_section_id_resolves(self) -> None:
        run_fiche_assertions(
            [fiche("F1", ["LEGISCTA000006157200", "LEGISCTA000099999999"])],
            [article_with_section("cid-1", "LEGISCTA000006157200")],
        )

    def test_rejects_a_fiche_where_no_section_id_resolves(self) -> None:
        with pytest.raises(CorpusAssertionError, match="assertion 5"):
            run_fiche_assertions(
                [fiche("F1", ["LEGISCTA000099999999"])],
                [article_with_section("cid-1", "LEGISCTA000006157200")],
            )

    def test_rejects_a_fiche_with_no_section_ids_at_all(self) -> None:
        with pytest.raises(CorpusAssertionError, match="assertion 5"):
            run_fiche_assertions(
                [fiche("F1", [])],
                [article_with_section("cid-1", "LEGISCTA000006157200")],
            )

    def test_a_partially_resolving_fiche_still_passes(self) -> None:
        """Not every `<dc:source>` id needs an article — only one, per SPEC §7.4."""
        run_fiche_assertions(
            [fiche("F1", ["LEGISCTA000006157200", "LEGISCTA000006999999"])],
            [article_with_section("cid-1", "LEGISCTA000006157200")],
        )


def test_reports_every_violation_at_once_not_just_the_first() -> None:
    """A refresh failing for unrelated reasons should say so in one run, not three."""
    with pytest.raises(CorpusAssertionError) as exc_info:
        run_article_assertions(
            [
                article("cid-1", "L113-3", etat="ABROGE_DIFF"),
                article("cid-1", None),
            ]
        )
    message = str(exc_info.value)
    assert "assertion 1" in message
    assert "assertion 2" in message
    assert "assertion 4" in message
