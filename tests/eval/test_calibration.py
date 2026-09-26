"""The judge calibration set and harness (`rag.eval.calibration`, SPEC §12.10, #47): the
`eval/calibration/judge-set.yaml` schema, its per-archetype validation, and the per-pair
report — detection and the direction of every error — against a fake judge."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rag.eval.calibration import (
    TARGET_METRIC,
    CalibrationAnswer,
    CalibrationPair,
    ErrorDirection,
    FaultArchetype,
    HumanLabel,
    JudgeSetError,
    dump_judge_set,
    load_calibration_run,
    load_judge_set,
    run_calibration,
    score_pair,
    summarize,
    validate_judge_set,
)
from rag.eval.judge import (
    FaithfulnessOutput,
    Judge,
    JudgedMetric,
    PointCoverageOutput,
    PointVerdict,
    PromptLanguage,
)
from rag.eval.schema import GoldenItem
from rag.generation.schema import FondementJuridique, Motif, Refus, Reponse

CONTEXT = "### Code des assurances\n\n— L113-2 · Titre\nL'assuré est obligé de payer la prime."


def _reponse(explanation: str = "Vous devez payer la prime.", cited: tuple[str, ...] = ("L113-2",)) -> Reponse:
    return Reponse(
        explanation=explanation,
        fondement_juridique=[FondementJuridique(article_id=a, gloss="obligations") for a in cited],
    )


def _pair(archetype: FaultArchetype = FaultArchetype.CLAIM_ABSENT_FROM_CONTEXT, **overrides: Any) -> CalibrationPair:
    fields: dict[str, Any] = {
        "golden_id": "gs-002",
        "archetype": archetype,
        "question": "Dois-je payer ma prime ?",
        "expected_points": ("l'assuré doit payer la prime",),
        "retrieved_citation_ids": ("L113-2",),
        "context": CONTEXT,
        "clean": CalibrationAnswer(_reponse(), HumanLabel.PASS),
        "faulted": CalibrationAnswer(
            _reponse("Vous devez payer la prime, sous peine de prison."), HumanLabel.FAIL
        ),
    }
    fields.update(overrides)
    return CalibrationPair(**fields)


def _golden(golden_id: str = "gs-002", **overrides: Any) -> GoldenItem:
    fields: dict[str, Any] = {
        "id": golden_id,
        "question": "Dois-je payer ma prime ?",
        "history": (),
        "expected_state": "reponse",
        "gold_fiches": (),
        "gold_spans": (),
        "gold_articles": (),
        "expected_points": ("l'assuré doit payer la prime",),
        "tags": (),
    }
    fields.update(overrides)
    return GoldenItem(**fields)


class TestJudgeSetRoundTrip:
    def test_dump_then_load_is_identity(self, tmp_path: Path) -> None:
        refusal_pair = _pair(
            FaultArchetype.REFUSAL_WITHOUT_EXPLANATION,
            golden_id="gs-010",
            clean=CalibrationAnswer(
                Refus(explanation="Je ne peux pas choisir pour vous. La loi prévoit ceci.", motif=Motif.CONSEIL_ACTION),
                HumanLabel.PASS,
            ),
            faulted=CalibrationAnswer(
                Refus(explanation="Je ne peux pas choisir pour vous.", motif=Motif.CONSEIL_ACTION), HumanLabel.FAIL
            ),
        )
        pairs = [_pair(), refusal_pair]
        path = tmp_path / "judge-set.yaml"

        dump_judge_set(pairs, path)

        assert load_judge_set(path) == pairs

    def test_is_written_as_reviewable_block_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "judge-set.yaml"
        dump_judge_set([_pair()], path)

        text = path.read_text(encoding="utf-8")
        assert text.startswith("- golden_id: gs-002\n")
        assert "archetype: claim_absent_from_context" in text
        assert "{" not in text

    def test_missing_file_is_an_empty_set(self, tmp_path: Path) -> None:
        assert load_judge_set(tmp_path / "absent.yaml") == []

    def test_shape_errors_are_all_reported_at_once(self, tmp_path: Path) -> None:
        path = tmp_path / "judge-set.yaml"
        path.write_text(
            "- golden_id: gs-001\n  archetype: nonsense\n- golden_id: gs-002\n", encoding="utf-8"
        )

        with pytest.raises(JudgeSetError) as excinfo:
            load_judge_set(path)

        message = str(excinfo.value)
        assert "gs-001" in message
        assert "gs-002" in message


class TestValidateJudgeSet:
    def test_a_well_formed_pair_passes(self) -> None:
        assert validate_judge_set([_pair()], [_golden()]) == []

    def test_golden_id_must_exist_in_the_golden_set(self) -> None:
        [violation] = validate_judge_set([_pair(golden_id="gs-999")], [_golden()])
        assert "gs-999" in violation

    def test_one_pair_per_golden_item(self) -> None:
        violations = validate_judge_set([_pair(), _pair()], [_golden()])
        assert any("more than one pair" in v for v in violations)

    def test_question_and_points_must_match_the_golden_item(self) -> None:
        violations = validate_judge_set([_pair(question="autre", expected_points=("x",))], [_golden()])
        assert any("question" in v for v in violations)
        assert any("expected_points" in v for v in violations)

    def test_the_twin_must_differ_from_its_clean_sibling(self) -> None:
        pair = _pair(faulted=CalibrationAnswer(_reponse(), HumanLabel.FAIL))
        [violation] = validate_judge_set([pair], [_golden()])
        assert "identical" in violation

    def test_fabricated_citation_must_cite_something_outside_the_retrieved_context(self) -> None:
        pair = _pair(
            FaultArchetype.FABRICATED_CITATION,
            faulted=CalibrationAnswer(_reponse(cited=("L113-2", "L113-3")), HumanLabel.FAIL),
        )
        assert validate_judge_set([pair], [_golden()]) == []

        not_fabricated = replace(pair, retrieved_citation_ids=("L113-2", "L113-3"))
        [violation] = validate_judge_set([not_fabricated], [_golden()])
        assert "fabricated" in violation

    def test_fabricated_citation_needs_a_clean_sibling_that_cites_validly(self) -> None:
        pair = _pair(
            FaultArchetype.FABRICATED_CITATION,
            clean=CalibrationAnswer(_reponse(cited=("L999",)), HumanLabel.PASS),
            faulted=CalibrationAnswer(_reponse(cited=("L999", "L998")), HumanLabel.FAIL),
        )
        violations = validate_judge_set([pair], [_golden()])
        assert any("clean" in v for v in violations)

    def test_dropped_point_needs_points_to_drop(self) -> None:
        pair = _pair(FaultArchetype.DROPPED_EXPECTED_POINT, expected_points=())
        violations = validate_judge_set([pair], [_golden(expected_points=())])
        assert any("expected_points" in v for v in violations)

    def test_refusal_archetype_needs_two_refusals_with_the_same_motif_and_a_shorter_twin(self) -> None:
        pair = _pair(FaultArchetype.REFUSAL_WITHOUT_EXPLANATION)
        violations = validate_judge_set([pair], [_golden()])
        assert any("Refus" in v for v in violations)

        longer_twin = _pair(
            FaultArchetype.REFUSAL_WITHOUT_EXPLANATION,
            clean=CalibrationAnswer(Refus(explanation="Non.", motif=Motif.CONSEIL_ACTION), HumanLabel.PASS),
            faulted=CalibrationAnswer(
                Refus(explanation="Non, et voici bien plus.", motif=Motif.CONSEIL_ACTION), HumanLabel.FAIL
            ),
        )
        violations = validate_judge_set([longer_twin], [_golden()])
        assert any("shorter" in v for v in violations)


class _ScoresByAnswer:
    """A fake judge call that scores an answer by looking up its explanation."""

    def __init__(self, scores: dict[str, float], provider: str = "Anthropic") -> None:
        self.scores = scores
        self.provider = provider

    def _score_for(self, prompt: str) -> float:
        # Longest explanation first: a twin's explanation often extends its clean sibling's.
        for explanation in sorted(self.scores, key=len, reverse=True):
            if explanation in prompt:
                return self.scores[explanation]
        raise AssertionError(f"no scripted score for prompt: {prompt[:200]}")

    def __call__(self, prompt: str, schema: type[Any]) -> tuple[Any, str]:
        score = self._score_for(prompt)
        if schema is FaithfulnessOutput:
            return FaithfulnessOutput(score=score, reasoning=f"scored {score}"), self.provider
        return (
            PointCoverageOutput(verdicts=[PointVerdict(index=1, asserted=score >= 1.0, reasoning="r")]),
            self.provider,
        )


def _judge(scores: dict[str, float], provider: str = "Anthropic") -> Judge:
    return Judge(
        call=_ScoresByAnswer(scores, provider),
        model="anthropic/claude-sonnet-5",
        faithfulness_language=PromptLanguage.EN,
        point_coverage_language=PromptLanguage.FR,
    )


CLEAN = "Vous devez payer la prime."
FAULTED = "Vous devez payer la prime, sous peine de prison."


class TestScorePair:
    def test_every_archetype_targets_one_judged_metric(self) -> None:
        assert set(TARGET_METRIC) == set(FaultArchetype)
        assert TARGET_METRIC[FaultArchetype.FABRICATED_CITATION] is JudgedMetric.FAITHFULNESS
        assert TARGET_METRIC[FaultArchetype.CLAIM_ABSENT_FROM_CONTEXT] is JudgedMetric.FAITHFULNESS
        assert TARGET_METRIC[FaultArchetype.DROPPED_EXPECTED_POINT] is JudgedMetric.POINT_COVERAGE
        assert TARGET_METRIC[FaultArchetype.REFUSAL_WITHOUT_EXPLANATION] is JudgedMetric.POINT_COVERAGE

    def test_detected_when_the_twin_scores_strictly_lower(self) -> None:
        result = score_pair(_judge({CLEAN: 1.0, FAULTED: 0.5}), _pair())

        assert result.metric is JudgedMetric.FAITHFULNESS
        assert result.clean_score == 1.0
        assert result.faulted_score == 0.5
        assert result.detected is True
        assert result.clean_error is None
        assert result.faulted_error is None

    def test_a_tie_is_not_a_detection(self) -> None:
        result = score_pair(_judge({CLEAN: 1.0, FAULTED: 1.0}), _pair())

        assert result.detected is False

    def test_a_passed_faulted_twin_is_a_false_pass(self) -> None:
        """The error this eval exists to catch (SPEC §12.10)."""
        result = score_pair(_judge({CLEAN: 1.0, FAULTED: 1.0}), _pair())

        assert result.faulted_error is ErrorDirection.FALSE_PASS
        assert result.clean_error is None

    def test_a_failed_clean_answer_is_a_false_fail(self) -> None:
        result = score_pair(_judge({CLEAN: 0.5, FAULTED: 0.5}), _pair())

        assert result.clean_error is ErrorDirection.FALSE_FAIL
        assert result.faulted_error is None

    def test_errors_are_measured_against_the_human_labels_not_assumed(self) -> None:
        """A twin the human judged still acceptable is not a false pass when it passes."""
        pair = _pair(faulted=CalibrationAnswer(_reponse(FAULTED), HumanLabel.PASS))

        result = score_pair(_judge({CLEAN: 1.0, FAULTED: 1.0}), pair)

        assert result.faulted_error is None

    def test_pass_threshold_is_configurable(self) -> None:
        result = score_pair(_judge({CLEAN: 0.9, FAULTED: 0.5}), _pair(), pass_threshold=0.8)

        assert result.clean_error is None
        assert result.faulted_error is None

    def test_point_coverage_archetypes_are_judged_on_coverage(self) -> None:
        pair = _pair(FaultArchetype.DROPPED_EXPECTED_POINT)

        result = score_pair(_judge({CLEAN: 1.0, FAULTED: 0.0}), pair)

        assert result.metric is JudgedMetric.POINT_COVERAGE
        assert result.detected is True

    def test_records_the_resolved_provider(self) -> None:
        result = score_pair(_judge({CLEAN: 1.0, FAULTED: 0.5}, provider="Google"), _pair())

        assert result.providers == ("Google",)


class TestSummarize:
    def _results(self, scores: list[tuple[float, float]]) -> list[Any]:
        results = []
        for index, (clean, faulted) in enumerate(scores):
            clean_text, faulted_text = f"Clean {index}.", f"Faulted {index}."
            pair = _pair(
                golden_id=f"gs-{index:03d}",
                clean=CalibrationAnswer(_reponse(clean_text), HumanLabel.PASS),
                faulted=CalibrationAnswer(_reponse(faulted_text), HumanLabel.FAIL),
            )
            results.append(score_pair(_judge({clean_text: clean, faulted_text: faulted}), pair))
        return results

    def test_ten_of_twelve_detected_with_no_leniency_passes(self) -> None:
        summary = summarize(self._results([(1.0, 0.5)] * 10 + [(0.5, 0.5)] * 2))

        assert summary.pairs == 12
        assert summary.detected == 10
        assert summary.detection_required == 10
        assert summary.false_passes == 0
        assert summary.false_fails == 2
        assert summary.systematically_lenient is False
        assert summary.passed is True

    def test_nine_of_twelve_fails_detection(self) -> None:
        summary = summarize(self._results([(1.0, 0.5)] * 9 + [(0.5, 0.5)] * 3))

        assert summary.passed is False

    def test_leniency_disqualifies_regardless_of_rate(self) -> None:
        """SPEC §12.10: a judge whose every error is a false pass is useless here."""
        summary = summarize(self._results([(1.0, 0.5)] * 11 + [(1.0, 1.0)]))

        assert summary.detected == 11
        assert summary.false_passes == 1
        assert summary.systematically_lenient is True
        assert summary.passed is False

    def test_the_detection_bar_scales_to_an_arbitrary_pair_count(self) -> None:
        assert summarize(self._results([(1.0, 0.5)] * 6)).detection_required == 5


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


class TestRunCalibration:
    def test_persists_per_pair_rows_and_a_header_pinning_the_judge(self, tmp_path: Path) -> None:
        _git(tmp_path, "init")
        _git(tmp_path, "config", "user.email", "t@example.com")
        _git(tmp_path, "config", "user.name", "T")
        judge_set_path = tmp_path / "judge-set.yaml"
        dump_judge_set([_pair()], judge_set_path)
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-m", "judge set")

        run = run_calibration(
            load_judge_set(judge_set_path),
            _judge({CLEAN: 1.0, FAULTED: 0.5}, provider="Anthropic"),
            run_id="calibration-test",
            repo_root=tmp_path,
            judge_set_path=judge_set_path,
            runs_dir=tmp_path / "runs",
        )

        assert run.header.judge_model == "anthropic/claude-sonnet-5"
        assert run.header.judge_provider == "Anthropic"
        assert run.header.judge_prompt_languages == {"faithfulness": "en", "point_coverage": "fr"}
        assert run.header.pass_threshold == 1.0
        assert len(run.header.judge_set_git_sha) == 40
        [row] = run.pairs
        assert row.golden_id == "gs-002"
        assert row.detected is True
        assert load_calibration_run(tmp_path / "runs" / "calibration-test.json") == run
