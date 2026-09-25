"""SPEC §12.9, ADR-0009, #46 — the three deterministic generation metrics: state accuracy,
citation validity, citation correctness."""

from __future__ import annotations

from rag.eval.generation_metrics import (
    citation_correctness,
    resolve_cited_cids,
    score_item,
    state_label,
)
from rag.eval.schema import GoldenItem
from rag.generation.citation import CitationOutcome
from rag.generation.pipeline import GenerationResult
from rag.generation.schema import FondementJuridique, Motif, Refus, Reponse
from rag.retrieval.candidates import Candidate, Provenance, Register


def golden_item(
    id: str = "gs-001",
    *,
    expected_state: str = "reponse",
    gold_articles: tuple[str, ...] = (),
) -> GoldenItem:
    return GoldenItem(
        id=id,
        question="une question",
        history=(),
        expected_state=expected_state,
        gold_fiches=(),
        gold_spans=(),
        gold_articles=gold_articles,
        expected_points=(),
        tags=(),
    )


def make_article(citation_id: str, legiarti_cid: str) -> Candidate:
    return Candidate(
        id=f"point-{legiarti_cid}",
        score=0.9,
        register=Register.ARTICLE,
        payload={"citation_id": citation_id, "legiarti_cid": legiarti_cid, "text": "..."},
        provenance=frozenset({Provenance.SEARCH}),
    )


def result_for(
    envelope: Reponse | Refus,
    *,
    cited: frozenset[str] = frozenset(),
    fabricated: frozenset[str] = frozenset(),
    contexts: list[Candidate] | None = None,
) -> GenerationResult:
    outcome = CitationOutcome(cited_ids=cited, retrieved_ids=cited - fabricated, fabricated_ids=fabricated)
    return GenerationResult(envelope=envelope, citation_outcome=outcome, contexts=contexts or [], candidate_pools={})


class TestStateLabel:
    def test_reponse_with_fondement_is_plain_reponse(self) -> None:
        result = result_for(Reponse(explanation="...", fondement_juridique=[]))
        assert state_label(result) == "reponse"

    def test_reponse_with_aucun_fondement_filled_is_reponse_sans_article(self) -> None:
        result = result_for(Reponse(explanation="...", aucun_fondement="rien trouvé"))
        assert state_label(result) == "reponse_sans_article"

    def test_refus_carries_its_motif(self) -> None:
        result = result_for(Refus(explanation="...", motif=Motif.HORS_CORPUS))
        assert state_label(result) == "refus:hors_corpus"

    def test_refus_conseil_action(self) -> None:
        result = result_for(Refus(explanation="...", motif=Motif.CONSEIL_ACTION))
        assert state_label(result) == "refus:conseil_action"

    def test_refus_recommandation_produit(self) -> None:
        result = result_for(Refus(explanation="...", motif=Motif.RECOMMANDATION_PRODUIT))
        assert state_label(result) == "refus:recommandation_produit"


class TestResolveCitedCids:
    def test_resolves_a_cited_citation_id_to_its_cid_via_the_retrieved_contexts(self) -> None:
        contexts = [make_article("L113-3", "CID1")]
        assert resolve_cited_cids(frozenset({"L113-3"}), contexts) == frozenset({"CID1"})

    def test_a_cited_id_absent_from_contexts_resolves_to_nothing(self) -> None:
        """A fabricated citation (SPEC §10.5) names no retrieved candidate to resolve
        through — already flagged by `citation_valid`, not double-counted here."""
        contexts = [make_article("L113-3", "CID1")]
        assert resolve_cited_cids(frozenset({"L999-9"}), contexts) == frozenset()

    def test_ignores_fiche_candidates(self) -> None:
        fiche = Candidate(
            id="fiche-1", score=0.9, register=Register.FICHE, payload={"fiche_id": "F1"}, provenance=frozenset()
        )
        assert resolve_cited_cids(frozenset({"F1"}), [fiche]) == frozenset()


class TestCitationCorrectness:
    def test_none_on_empty_gold_articles(self) -> None:
        assert citation_correctness(frozenset({"CID1"}), frozenset()) is None

    def test_full_overlap_is_one(self) -> None:
        assert citation_correctness(frozenset({"CID1", "CID2"}), frozenset({"CID1", "CID2"})) == 1.0

    def test_no_overlap_is_zero(self) -> None:
        assert citation_correctness(frozenset({"CID9"}), frozenset({"CID1"})) == 0.0

    def test_partial_overlap_is_a_fraction_of_gold(self) -> None:
        # Cited one of two gold articles, plus an extra not in gold — the extra never
        # inflates the score, since the denominator is |gold|, not |cited|.
        assert citation_correctness(frozenset({"CID1", "CID9"}), frozenset({"CID1", "CID2"})) == 0.5


class TestScoreItem:
    def test_state_correct_when_labels_match(self) -> None:
        item = golden_item(expected_state="reponse")
        result = result_for(Reponse(explanation="...", fondement_juridique=[]))

        score = score_item(item, result)

        assert score.item_id == "gs-001"
        assert score.expected_state == "reponse"
        assert score.actual_state == "reponse"
        assert score.state_correct is True

    def test_state_incorrect_when_labels_disagree(self) -> None:
        item = golden_item(expected_state="refus:hors_corpus")
        result = result_for(Reponse(explanation="...", fondement_juridique=[]))

        score = score_item(item, result)

        assert score.state_correct is False

    def test_citation_valid_reads_straight_off_the_generation_result(self) -> None:
        item = golden_item(gold_articles=("CID1",))
        result = result_for(
            Reponse(explanation="...", fondement_juridique=[FondementJuridique(article_id="L999-9", gloss="x")]),
            cited=frozenset({"L999-9"}),
            fabricated=frozenset({"L999-9"}),
        )

        score = score_item(item, result)

        assert score.citation_valid is False
        assert score.fabricated_ids == ("L999-9",)

    def test_citation_correctness_is_computed_against_gold_articles_via_the_cited_id_join(self) -> None:
        item = golden_item(gold_articles=("CID1", "CID2"))
        result = result_for(
            Reponse(explanation="...", fondement_juridique=[FondementJuridique(article_id="L113-3", gloss="x")]),
            cited=frozenset({"L113-3"}),
            contexts=[make_article("L113-3", "CID1")],
        )

        score = score_item(item, result)

        assert score.citation_correctness == 0.5

    def test_citation_correctness_is_none_when_no_gold_articles(self) -> None:
        item = golden_item(expected_state="refus:hors_corpus", gold_articles=())
        result = result_for(Refus(explanation="...", motif=Motif.HORS_CORPUS))

        score = score_item(item, result)

        assert score.citation_correctness is None

    def test_never_scores_a_repaired_envelope(self) -> None:
        """SPEC §10.5: eval always sees `generate()`'s unrepaired output — `score_item`
        itself has no repair step, so this is really pinning that `citation_outcome`, not
        some display-repaired copy of the envelope, is what feeds every field here."""
        item = golden_item(gold_articles=("CID1",))
        result = result_for(
            Reponse(explanation="...", fondement_juridique=[FondementJuridique(article_id="L999-9", gloss="x")]),
            cited=frozenset({"L999-9"}),
            fabricated=frozenset({"L999-9"}),
        )

        score = score_item(item, result)

        assert score.citation_valid is False
        assert score.citation_correctness == 0.0
