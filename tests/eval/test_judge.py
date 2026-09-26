"""The two judged metrics (`rag.eval.judge`, SPEC §12.9/§12.10, #47) against a fake
`JudgeCall` — the prompt the judge is handed and the score computed from what it returns,
never a live OpenRouter call (`rag.eval.judge_chain` is the real seam)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from rag.eval.judge import (
    FAITHFULNESS_V2_PROMPT_EN,
    FaithfulnessOutput,
    Judge,
    JudgedMetric,
    JudgeOutputError,
    PointCoverageOutput,
    PointVerdict,
    PromptLanguage,
    render_answer,
)
from rag.generation.schema import FondementJuridique, Motif, Refus, Reponse


class _FakeCall:
    """Records every prompt it is handed and answers from a queue of canned outputs."""

    def __init__(self, *outputs: BaseModel, provider: str = "Anthropic") -> None:
        self.outputs = list(outputs)
        self.provider = provider
        self.calls: list[tuple[str, type[BaseModel]]] = []

    def __call__(self, prompt: str, schema: type[Any]) -> tuple[Any, str]:
        self.calls.append((prompt, schema))
        output = self.outputs.pop(0)
        assert isinstance(output, schema)
        return output, self.provider


def _judge(call: _FakeCall, **overrides: Any) -> Judge:
    fields: dict[str, Any] = {
        "call": call,
        "model": "anthropic/claude-sonnet-5",
        "faithfulness_language": PromptLanguage.EN,
        "point_coverage_language": PromptLanguage.FR,
    }
    fields.update(overrides)
    return Judge(**fields)


class TestRenderAnswer:
    """The judge sees the whole envelope, not just `explanation` — a fabricated citation
    lives in `fondement_juridique`, so an explanation-only rendering could never catch it."""

    def test_reponse_carries_explanation_and_every_citation(self) -> None:
        envelope = Reponse(
            explanation="Le locataire doit s'assurer.",
            fondement_juridique=[FondementJuridique(article_id="L113-2", gloss="obligations de l'assuré")],
        )

        rendered = render_answer(envelope)

        assert "Le locataire doit s'assurer." in rendered
        assert "L113-2" in rendered
        assert "obligations de l'assuré" in rendered

    def test_reponse_sans_article_states_the_absence(self) -> None:
        envelope = Reponse(explanation="Aucune règle précise.", aucun_fondement="Pas d'article applicable.")

        assert "Pas d'article applicable." in render_answer(envelope)

    def test_refus_names_its_motif(self) -> None:
        envelope = Refus(explanation="Je ne peux pas recommander.", motif=Motif.RECOMMANDATION_PRODUIT)

        rendered = render_answer(envelope)

        assert "recommandation_produit" in rendered
        assert "Je ne peux pas recommander." in rendered


class TestFaithfulness:
    def test_english_prompt_is_managed_faithfulness_v2_verbatim_with_its_variables_filled(self) -> None:
        call = _FakeCall(FaithfulnessOutput(score=0.75, reasoning="3 of 4 supported"))

        _judge(call).faithfulness(context="CONTEXTE {{x}}", answer="RÉPONSE")

        [(prompt, schema)] = call.calls
        assert schema is FaithfulnessOutput
        assert prompt == FAITHFULNESS_V2_PROMPT_EN.replace("{{context}}", "CONTEXTE {{x}}").replace(
            "{{answer}}", "RÉPONSE"
        )

    def test_french_prompt_is_selectable(self) -> None:
        call = _FakeCall(FaithfulnessOutput(score=1.0, reasoning="ok"))

        _judge(call, faithfulness_language=PromptLanguage.FR).faithfulness(context="C", answer="A")

        [(prompt, _)] = call.calls
        assert prompt != FAITHFULNESS_V2_PROMPT_EN.replace("{{context}}", "C").replace("{{answer}}", "A")
        assert "Contexte" in prompt

    def test_returns_the_score_reasoning_and_resolved_provider(self) -> None:
        call = _FakeCall(FaithfulnessOutput(score=0.5, reasoning="half"), provider="Google")

        score = _judge(call).faithfulness(context="C", answer="A")

        assert score.metric is JudgedMetric.FAITHFULNESS
        assert score.value == 0.5
        assert score.reasoning == "half"
        assert score.provider == "Google"

    def test_output_schema_bounds_the_score_to_a_proportion(self) -> None:
        with pytest.raises(ValueError):
            FaithfulnessOutput(score=1.5, reasoning="")


class TestPointCoverage:
    def test_score_is_computed_from_per_point_verdicts_not_asked_of_the_judge(self) -> None:
        call = _FakeCall(
            PointCoverageOutput(
                verdicts=[
                    PointVerdict(index=1, asserted=True, reasoning="dit"),
                    PointVerdict(index=2, asserted=False, reasoning="absent"),
                    PointVerdict(index=3, asserted=True, reasoning="dit"),
                ]
            )
        )

        score = _judge(call).point_coverage(question="Q", answer="A", expected_points=("p1", "p2", "p3"))

        assert score is not None
        assert score.metric is JudgedMetric.POINT_COVERAGE
        assert score.value == pytest.approx(2 / 3)
        assert "2: absent" in score.reasoning

    def test_prompt_numbers_every_expected_point(self) -> None:
        call = _FakeCall(PointCoverageOutput(verdicts=[PointVerdict(index=1, asserted=True, reasoning="")]))

        _judge(call).point_coverage(question="Ma question ?", answer="Ma réponse.", expected_points=("le point",))

        [(prompt, schema)] = call.calls
        assert schema is PointCoverageOutput
        assert "1. le point" in prompt
        assert "Ma question ?" in prompt
        assert "Ma réponse." in prompt

    def test_both_languages_have_distinct_prompts(self) -> None:
        outputs = [PointCoverageOutput(verdicts=[PointVerdict(index=1, asserted=True, reasoning="")]) for _ in range(2)]
        call = _FakeCall(*outputs)

        _judge(call, point_coverage_language=PromptLanguage.FR).point_coverage(
            question="Q", answer="A", expected_points=("p",)
        )
        _judge(call, point_coverage_language=PromptLanguage.EN).point_coverage(
            question="Q", answer="A", expected_points=("p",)
        )

        [(fr_prompt, _), (en_prompt, _)] = call.calls
        assert fr_prompt != en_prompt

    def test_no_expected_points_is_undefined_not_zero_and_costs_no_call(self) -> None:
        """`hors_corpus` items carry no points (SPEC §12.9) — nothing to cover."""
        call = _FakeCall()

        assert _judge(call).point_coverage(question="Q", answer="A", expected_points=()) is None
        assert call.calls == []

    @pytest.mark.parametrize(
        "indices",
        [(1,), (1, 2, 2), (1, 3), (0, 1)],
        ids=["missing", "duplicate", "unknown", "zero-based"],
    )
    def test_a_verdict_set_that_does_not_match_the_points_is_rejected(self, indices: tuple[int, ...]) -> None:
        """A silently skipped point would read as not asserted — a judge malfunction must
        not be scored as an answer failure."""
        call = _FakeCall(
            PointCoverageOutput(verdicts=[PointVerdict(index=i, asserted=True, reasoning="") for i in indices])
        )

        with pytest.raises(JudgeOutputError):
            _judge(call).point_coverage(question="Q", answer="A", expected_points=("p1", "p2"))


def test_languages_are_reported_per_metric() -> None:
    judge = _judge(_FakeCall(), faithfulness_language=PromptLanguage.FR, point_coverage_language=PromptLanguage.EN)

    assert judge.languages() == {"faithfulness": "fr", "point_coverage": "en"}
