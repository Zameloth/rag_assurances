"""`rag.app.view.build_view` — what the app renders from one `GenerationResult` (SPEC §13.5,
§10.5, #50)."""

from __future__ import annotations

from datetime import date

from rag.app.view import AnswerState, build_view, legifrance_url
from rag.generation.citation import check_citations
from rag.generation.pipeline import GenerationResult
from rag.generation.schema import Envelope, FondementJuridique, Motif, Refus, Reponse
from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.quota import NO_ARTICLE_MARKER_ID, NO_ARTICLE_MARKER_TEXT


def _result(envelope: Envelope, contexts: list[Candidate]) -> GenerationResult:
    return GenerationResult(
        envelope=envelope,
        citation_outcome=check_citations(envelope, contexts),
        contexts=contexts,
    )


def _fiche(fiche_id: str, chunk_index: int = 0) -> Candidate:
    return Candidate(
        id=f"{fiche_id}#{chunk_index}",
        score=0.9,
        register=Register.FICHE,
        payload={
            "fiche_id": fiche_id,
            "chunk_index": chunk_index,
            "title": f"Titre {fiche_id}",
            "sp_url": f"https://www.service-public.gouv.fr/particuliers/vosdroits/{fiche_id}",
            "date_modified": "2026-07-15",
            "text": "...",
        },
        provenance=frozenset({Provenance.SEARCH}),
    )


def no_article_marker() -> Candidate:
    return Candidate(
        id=NO_ARTICLE_MARKER_ID,
        score=0.0,
        register=Register.ARTICLE,
        payload={"text": NO_ARTICLE_MARKER_TEXT},
        provenance=frozenset(),
    )


L113_12 = Candidate(
    id="point-L113-12",
    score=0.9,
    register=Register.ARTICLE,
    payload={
        "citation_id": "L113-12",
        "legiarti_cid": "LEGIARTI000006792938",
        "legiarti_version_id": "LEGIARTI000041378467",
        "text": "...",
    },
    provenance=frozenset({Provenance.SEARCH}),
)


class TestState:
    def test_a_reponse_with_articles_is_answerable(self) -> None:
        envelope = Reponse(
            explanation="...", fondement_juridique=[FondementJuridique(article_id="L113-12", gloss="g")]
        )
        assert build_view(_result(envelope, [L113_12])).state is AnswerState.REPONSE

    def test_a_reponse_with_aucun_fondement_is_no_article(self) -> None:
        envelope = Reponse(explanation="...", aucun_fondement=NO_ARTICLE_MARKER_TEXT)
        view = build_view(_result(envelope, [no_article_marker()]))
        assert view.state is AnswerState.REPONSE_SANS_ARTICLE

    def test_both_regulated_act_motifs_are_one_state(self) -> None:
        for motif in (Motif.RECOMMANDATION_PRODUIT, Motif.CONSEIL_ACTION):
            envelope = Refus(explanation="...", motif=motif)
            assert build_view(_result(envelope, [])).state is AnswerState.REFUS_REGULE

    def test_hors_corpus_is_its_own_state(self) -> None:
        envelope = Refus(explanation="...", motif=Motif.HORS_CORPUS)
        assert build_view(_result(envelope, [])).state is AnswerState.HORS_CORPUS


class TestCitations:
    def test_article_links_to_the_version_id_never_the_cid(self) -> None:
        """SPEC §13.5: `cid` points at the article's *first* version — superseded text under
        a citation just called in-force."""
        envelope = Reponse(
            explanation="...",
            fondement_juridique=[FondementJuridique(article_id="L113-12", gloss="résiliation annuelle")],
        )
        [citation] = build_view(_result(envelope, [L113_12])).fondement_juridique
        assert citation.citation_id == "L113-12"
        assert citation.gloss == "résiliation annuelle"
        assert citation.url == "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000041378467"
        assert "LEGIARTI000006792938" not in citation.url

    def test_legifrance_url_shape(self) -> None:
        assert legifrance_url("LEGIARTI0001") == "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI0001"

    def test_a_fabricated_citation_is_dropped_and_surfaces_the_no_article_marker(self) -> None:
        """SPEC §10.5: the demo drops the offending citation and surfaces the no-article
        marker — the display copy only, the result itself stays unrepaired."""
        envelope = Reponse(
            explanation="...",
            fondement_juridique=[FondementJuridique(article_id="L999-1", gloss="inventé")],
        )
        result = _result(envelope, [L113_12])
        view = build_view(result)

        assert view.fondement_juridique == ()
        assert view.state is AnswerState.REPONSE_SANS_ARTICLE
        assert isinstance(view.envelope, Reponse)
        assert view.envelope.aucun_fondement == NO_ARTICLE_MARKER_TEXT
        assert result.envelope.fondement_juridique[0].article_id == "L999-1"

    def test_refusals_keep_their_citations(self) -> None:
        """SPEC §10.4: refusals still answer the informational part."""
        envelope = Refus(
            explanation="...",
            motif=Motif.CONSEIL_ACTION,
            fondement_juridique=[FondementJuridique(article_id="L113-12", gloss="g")],
        )
        [citation] = build_view(_result(envelope, [L113_12])).fondement_juridique
        assert citation.citation_id == "L113-12"


class TestFiches:
    def test_fiche_sources_link_the_stored_sp_url_with_date_modified(self) -> None:
        envelope = Reponse(explanation="...", aucun_fondement=NO_ARTICLE_MARKER_TEXT)
        [source] = build_view(_result(envelope, [_fiche("F2594")])).fiches
        assert source.title == "Titre F2594"
        assert source.url == "https://www.service-public.gouv.fr/particuliers/vosdroits/F2594"
        assert source.date_modified == date(2026, 7, 15)

    def test_several_chunks_of_one_fiche_are_one_source_in_retrieval_order(self) -> None:
        contexts = [_fiche("F2594", 0), _fiche("F1124", 0), _fiche("F2594", 3)]
        envelope = Reponse(explanation="...", aucun_fondement=NO_ARTICLE_MARKER_TEXT)
        fiches = build_view(_result(envelope, contexts)).fiches
        assert [f.title for f in fiches] == ["Titre F2594", "Titre F1124"]

    def test_articles_and_the_marker_are_not_fiche_sources(self) -> None:
        envelope = Reponse(explanation="...", aucun_fondement=NO_ARTICLE_MARKER_TEXT)
        view = build_view(_result(envelope, [L113_12, no_article_marker()]))
        assert view.fiches == ()
