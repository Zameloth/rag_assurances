"""`rag.eval.propose`'s pure functions (#70, ADR-0020) — no LLM client, no corpus I/O, so
these are exercised directly against small synthetic fixtures rather than the committed
corpus (`test_eval_corpus.py`'s fixture is for that).
"""

from __future__ import annotations

import inspect

import rag.eval.propose as propose_module
from rag.eval.propose import filter_verbatim_spans, lexical_shortlist


def test_propose_module_never_imports_the_retrieval_pipeline() -> None:
    """ADR-0020's central constraint, checked statically against the actual import
    statements (not the prose discussing them): the shortlist algorithm must never import
    `rag.retrieval` (the embedder, reranker, or any ladder arm) — deriving gold labels
    from the pipeline under test would make rung 3 unfalsifiable in aggregate, the same
    failure already rejected once for `<dc:source>`."""
    import_lines = [
        line for line in inspect.getsource(propose_module).splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any("rag.retrieval" in line for line in import_lines)


def _article(cid: str, texte: str) -> dict[str, str]:
    return {"cid": cid, "citation_id": cid, "texte": texte}


# --- lexical_shortlist -------------------------------------------------------


def test_lexical_shortlist_ranks_the_article_matching_the_question_first() -> None:
    articles = [
        _article("A1", "Le contrat de résiliation prévoit un préavis de deux mois."),
        _article("A2", "La garantie dégât des eaux couvre les infiltrations."),
    ]
    result = lexical_shortlist("comment résilier mon contrat ?", "texte de fiche sans rapport", articles)
    assert result[0] == "A1"


def test_lexical_shortlist_respects_top_n() -> None:
    articles = [_article(f"A{i}", "assurance habitation locataire") for i in range(5)]
    result = lexical_shortlist("assurance habitation", "locataire", articles, top_n=2)
    assert len(result) == 2


def test_lexical_shortlist_excludes_articles_with_no_term_overlap() -> None:
    articles = [
        _article("A1", "dégât des eaux infiltration garantie"),
        _article("A2", "vocabulaire totalement disjoint xyzzy plugh"),
    ]
    result = lexical_shortlist("dégât des eaux", "infiltration garantie", articles)
    assert result == ["A1"]


def test_lexical_shortlist_returns_empty_for_blank_query_and_body() -> None:
    articles = [_article("A1", "peu importe le contenu")]
    assert lexical_shortlist("", "", articles) == []


def test_lexical_shortlist_weighs_the_question_higher_than_the_body() -> None:
    # "résiliation" only in the question; both articles share the body's vocabulary
    # equally, so only the question term should break the tie.
    articles = [
        _article("A1", "résiliation du contrat en cas de non-paiement"),
        _article("A2", "non-paiement du contrat entraîne une mise en demeure"),
    ]
    result = lexical_shortlist("résiliation", "contrat non-paiement", articles, top_n=1)
    assert result == ["A1"]


# --- filter_verbatim_spans ----------------------------------------------------


def test_filter_verbatim_spans_keeps_only_exact_substrings() -> None:
    chunks = ["l'assurance responsabilité civile est obligatoire pour le locataire."]
    candidates = [
        "l'assurance responsabilité civile est obligatoire",  # real substring
        "l'assurance est facultative pour le locataire",  # not present verbatim
    ]
    assert filter_verbatim_spans(candidates, chunks) == [candidates[0]]


def test_filter_verbatim_spans_drops_empty_and_deduplicates() -> None:
    chunks = ["le préavis est de deux mois."]
    candidates = ["le préavis est de deux mois.", "", "le préavis est de deux mois."]
    assert filter_verbatim_spans(candidates, chunks) == ["le préavis est de deux mois."]


def test_filter_verbatim_spans_checks_across_all_given_chunks() -> None:
    chunks = ["premier chunk sans le passage.", "second chunk avec le passage recherché."]
    assert filter_verbatim_spans(["le passage recherché"], chunks) == ["le passage recherché"]
