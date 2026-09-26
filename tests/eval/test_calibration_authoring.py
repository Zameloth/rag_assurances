"""The faulted-twin authoring helper's library half (`rag.eval.calibration_authoring`,
SPEC §12.10, #47) — drafting a pair from a real pipeline answer, the envelope's editable
YAML form, and the validated write into `judge-set.yaml`."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.eval.calibration import (
    CalibrationAnswer,
    CalibrationSetError,
    FaultArchetype,
    HumanLabel,
    load_calibration_set,
)
from rag.eval.calibration_authoring import (
    ARCHETYPE_GUIDANCE,
    draft_pair,
    envelope_from_yaml,
    envelope_to_yaml,
    parse_label,
    upsert_pair,
)
from rag.eval.schema import GoldenItem
from rag.generation.citation import check_citations
from rag.generation.pipeline import GenerationResult
from rag.generation.schema import FondementJuridique, Motif, Refus, Reponse
from rag.retrieval.candidates import Candidate, Provenance, Register


def _golden(golden_id: str = "gs-002") -> GoldenItem:
    return GoldenItem(
        id=golden_id,
        question="Dois-je payer ma prime ?",
        history=(),
        expected_state="reponse",
        gold_fiches=(),
        gold_spans=(),
        gold_articles=(),
        expected_points=("l'assuré doit payer la prime",),
        tags=(),
    )


def _article(citation_id: str, text: str) -> Candidate:
    return Candidate(
        id=f"pt-{citation_id}",
        register=Register.ARTICLE,
        score=1.0,
        payload={"citation_id": citation_id, "legiarti_cid": f"CID-{citation_id}", "text": text},
        provenance=frozenset({Provenance.SEARCH}),
    )


def _result() -> GenerationResult:
    contexts = [_article("L113-2", "L'assuré est obligé de payer la prime.")]
    envelope = Reponse(
        explanation="Vous devez payer la prime.",
        fondement_juridique=[FondementJuridique(article_id="L113-2", gloss="obligations")],
    )
    return GenerationResult(envelope=envelope, citation_outcome=check_citations(envelope, contexts), contexts=contexts)


def test_every_archetype_has_guidance() -> None:
    assert set(ARCHETYPE_GUIDANCE) == set(FaultArchetype)


class TestDraftPair:
    def test_clean_side_is_the_real_answer_labelled_pass(self) -> None:
        result = _result()

        pair = draft_pair(_golden(), result, FaultArchetype.CLAIM_ABSENT_FROM_CONTEXT)

        assert pair.clean == CalibrationAnswer(result.envelope, HumanLabel.PASS)

    def test_faulted_side_starts_as_a_copy_labelled_fail_for_the_author_to_edit(self) -> None:
        pair = draft_pair(_golden(), _result(), FaultArchetype.CLAIM_ABSENT_FROM_CONTEXT)

        assert pair.faulted.envelope == pair.clean.envelope
        assert pair.faulted.human_label is HumanLabel.FAIL

    def test_carries_the_item_and_what_the_generator_was_shown(self) -> None:
        pair = draft_pair(_golden(), _result(), FaultArchetype.FABRICATED_CITATION)

        assert pair.golden_id == "gs-002"
        assert pair.archetype is FaultArchetype.FABRICATED_CITATION
        assert pair.question == "Dois-je payer ma prime ?"
        assert pair.expected_points == ("l'assuré doit payer la prime",)
        assert pair.retrieved_citation_ids == ("L113-2",)
        assert "L'assuré est obligé de payer la prime." in pair.context
        assert "L113-2" in pair.context


class TestEnvelopeYaml:
    @pytest.mark.parametrize(
        "envelope",
        [
            Reponse(explanation="Oui.\nSur deux lignes.", fondement_juridique=[FondementJuridique(article_id="L1", gloss="g")]),
            Reponse(explanation="Non.", aucun_fondement="Aucun article."),
            Refus(explanation="Je ne peux pas.", motif=Motif.HORS_CORPUS),
        ],
        ids=["reponse", "reponse_sans_article", "refus"],
    )
    def test_round_trips(self, envelope: Reponse | Refus) -> None:
        assert envelope_from_yaml(envelope_to_yaml(envelope)) == envelope

    def test_an_edit_that_breaks_the_envelope_is_rejected_with_a_reason(self) -> None:
        with pytest.raises(CalibrationSetError, match="motif"):
            envelope_from_yaml("type: refus\nexplanation: Non.\n")


class TestUpsertPair:
    def _faulted(self) -> CalibrationAnswer:
        return CalibrationAnswer(Reponse(explanation="Vous devez payer la prime, sous peine de prison."), HumanLabel.FAIL)

    def test_appends_a_valid_pair(self, tmp_path: Path) -> None:
        path = tmp_path / "judge-set.yaml"
        pair = draft_pair(_golden(), _result(), FaultArchetype.CLAIM_ABSENT_FROM_CONTEXT)
        pair = pair.with_faulted(self._faulted())

        upsert_pair(path, pair, [_golden()])

        assert load_calibration_set(path) == [pair]

    def test_replaces_an_existing_pair_for_the_same_golden_item(self, tmp_path: Path) -> None:
        path = tmp_path / "judge-set.yaml"
        first = draft_pair(_golden(), _result(), FaultArchetype.CLAIM_ABSENT_FROM_CONTEXT).with_faulted(self._faulted())
        upsert_pair(path, first, [_golden()])
        second = first.with_faulted(CalibrationAnswer(Reponse(explanation="Autre faute."), HumanLabel.FAIL))

        upsert_pair(path, second, [_golden()])

        assert load_calibration_set(path) == [second]

    def test_an_invalid_pair_is_never_written(self, tmp_path: Path) -> None:
        path = tmp_path / "judge-set.yaml"
        untouched_twin = draft_pair(_golden(), _result(), FaultArchetype.CLAIM_ABSENT_FROM_CONTEXT)

        with pytest.raises(CalibrationSetError, match="identical"):
            upsert_pair(path, untouched_twin, [_golden()])

        assert not path.exists()


class TestParseLabel:
    """The author labels both answers (SPEC §12.10: "all human-labelled"); the draft's
    pass/fail is only the default offered, never recorded without being confirmed."""

    @pytest.mark.parametrize("raw", ["pass", "PASS", " p ", "Pass"])
    def test_accepts_pass(self, raw: str) -> None:
        assert parse_label(raw, default=HumanLabel.FAIL) is HumanLabel.PASS

    @pytest.mark.parametrize("raw", ["fail", "F", " fail "])
    def test_accepts_fail(self, raw: str) -> None:
        assert parse_label(raw, default=HumanLabel.PASS) is HumanLabel.FAIL

    def test_empty_takes_the_offered_default(self) -> None:
        assert parse_label("  ", default=HumanLabel.FAIL) is HumanLabel.FAIL

    def test_anything_else_is_not_a_label(self) -> None:
        assert parse_label("maybe", default=HumanLabel.PASS) is None
