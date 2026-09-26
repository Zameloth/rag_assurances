"""The judge calibration set and harness (SPEC §12.10, ADR-0025, #47).

**The calibration set is built, not sampled.** Sampled from normal pipeline output, ~18 of
20 answers are fine, judge and human both say "fine", agreement reads 95% and nothing is
learned — the judge's whole job is catching the *rare* failure. So each pair keeps one real
pipeline answer and a hand-authored **faulted twin** planting exactly one of four faults
(`FaultArchetype`). Pairing does the ladder's work: item difficulty cancels, and the test
is purely *does the judge score the faulted twin strictly below its clean sibling*, on the
one judged metric the planted fault targets (`TARGET_METRIC`).

**A pair is self-contained.** It carries the question, the `expected_points`, the rendered
context the generator was shown and the citation ids it could cite — not a pointer to
re-retrieve them. The set outlives its first use as a judge regression test (SPEC §12.10),
and a regression test whose inputs move with the index cannot tell "the instrument moved"
from "the fixture moved".

**Two bars, and the second matters more** (SPEC §12.10): detection on ≥ 10 of 12 pairs
(scaled for an arbitrary pair count), and no systematic leniency — measured against each
answer's *human label*, not assumed from which side of the pair it sits on. An answer the
judge passes (score ≥ `pass_threshold`) that the human failed is a **false pass**; the
reverse is a **false fail**. A judge whose errors are mostly false passes is disqualified
whatever its detection rate, because false passes are exactly what this eval exists to
catch.
"""

from __future__ import annotations

import enum
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from rag.eval.judge import Judge, JudgedMetric, JudgeScore, render_answer
from rag.eval.retrieval_run import resolve_git_sha
from rag.eval.schema import GoldenItem
from rag.generation.schema import Envelope, Refus

__all__ = [
    "DEFAULT_PASS_THRESHOLD",
    "DETECTION_BAR",
    "ENVELOPE_ADAPTER",
    "BlockStyleDumper",
    "TARGET_METRIC",
    "CalibrationAnswer",
    "CalibrationHeader",
    "CalibrationPair",
    "CalibrationRun",
    "CalibrationSummary",
    "ErrorDirection",
    "FaultArchetype",
    "HumanLabel",
    "JudgeSetError",
    "PairResult",
    "dump_judge_set",
    "load_calibration_run",
    "load_judge_set",
    "pair_violations",
    "run_calibration",
    "score_pair",
    "summarize",
    "validate_judge_set",
]

# SPEC §12.10 — "the judge scores the faulted twin strictly lower on ≥ 10 of 12 pairs".
DETECTION_BAR = (10, 12)

# Both human labels are binary, so the judge's continuous score has to be read as binary
# too before a false pass/fail can be counted. 1.0 is the reading the labels themselves
# use: a twin is labelled "fail" because *one* thing in it is wrong, so an answer passes
# only with every statement grounded / every point asserted.
DEFAULT_PASS_THRESHOLD = 1.0

ENVELOPE_ADAPTER: TypeAdapter[Envelope] = TypeAdapter(Envelope)


class FaultArchetype(enum.StrEnum):
    """SPEC §12.10's four planted faults."""

    FABRICATED_CITATION = "fabricated_citation"
    CLAIM_ABSENT_FROM_CONTEXT = "claim_absent_from_context"
    DROPPED_EXPECTED_POINT = "dropped_expected_point"
    REFUSAL_WITHOUT_EXPLANATION = "refusal_without_explanation"


# The judged metric each fault should move. A fabricated citation and an invented claim
# are both statements the context does not support; a dropped point and a refusal that
# omits its explaining half (SPEC §10.4) both leave an `expected_point` unasserted —
# refusal items carry points precisely so that a bare refusal fails (SPEC §12.9).
TARGET_METRIC: Mapping[FaultArchetype, JudgedMetric] = {
    FaultArchetype.FABRICATED_CITATION: JudgedMetric.FAITHFULNESS,
    FaultArchetype.CLAIM_ABSENT_FROM_CONTEXT: JudgedMetric.FAITHFULNESS,
    FaultArchetype.DROPPED_EXPECTED_POINT: JudgedMetric.POINT_COVERAGE,
    FaultArchetype.REFUSAL_WITHOUT_EXPLANATION: JudgedMetric.POINT_COVERAGE,
}


class HumanLabel(enum.StrEnum):
    PASS = "pass"
    FAIL = "fail"


class ErrorDirection(enum.StrEnum):
    FALSE_PASS = "false_pass"
    FALSE_FAIL = "false_fail"


class JudgeSetError(Exception):
    """`judge-set.yaml` doesn't shape into `CalibrationPair`s, or a pair breaks its
    archetype's rules. Raised with every violation found, not just the first."""


@dataclass(frozen=True)
class CalibrationAnswer:
    envelope: Envelope
    human_label: HumanLabel

    def as_dict(self) -> dict[str, Any]:
        return {"human_label": self.human_label.value, "envelope": ENVELOPE_ADAPTER.dump_python(self.envelope, mode="json")}


@dataclass(frozen=True)
class CalibrationPair:
    golden_id: str
    archetype: FaultArchetype
    question: str
    expected_points: tuple[str, ...]
    retrieved_citation_ids: tuple[str, ...]
    context: str
    clean: CalibrationAnswer
    faulted: CalibrationAnswer

    def with_faulted(self, faulted: CalibrationAnswer) -> CalibrationPair:
        return replace(self, faulted=faulted)

    def as_dict(self) -> dict[str, Any]:
        return {
            "golden_id": self.golden_id,
            "archetype": self.archetype.value,
            "question": self.question,
            "expected_points": list(self.expected_points),
            "retrieved_citation_ids": list(self.retrieved_citation_ids),
            "context": self.context,
            "clean": self.clean.as_dict(),
            "faulted": self.faulted.as_dict(),
        }


# --- the file --------------------------------------------------------------------------


def _parse_answer(raw: Any, where: str) -> CalibrationAnswer:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping with human_label and envelope")
    try:
        label = HumanLabel(str(raw.get("human_label")))
    except ValueError:
        raise ValueError(f"{where}.human_label: expected one of {[h.value for h in HumanLabel]}") from None
    try:
        envelope = ENVELOPE_ADAPTER.validate_python(raw.get("envelope"))
    except ValidationError as error:
        raise ValueError(f"{where}.envelope: {error.errors()[0]['msg']}") from None
    return CalibrationAnswer(envelope=envelope, human_label=label)


def _string_tuple(raw: Any, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise ValueError(f"{where}: expected a list of strings")
    return tuple(raw)


def _parse_pair(raw: Any) -> CalibrationPair:
    if not isinstance(raw, Mapping):
        raise ValueError("expected a mapping")
    for key in ("golden_id", "question", "context"):
        if not isinstance(raw.get(key), str):
            raise ValueError(f"{key}: expected a string")
    try:
        archetype = FaultArchetype(str(raw.get("archetype")))
    except ValueError:
        raise ValueError(f"archetype: expected one of {[a.value for a in FaultArchetype]}") from None
    return CalibrationPair(
        golden_id=raw["golden_id"],
        archetype=archetype,
        question=raw["question"],
        expected_points=_string_tuple(raw.get("expected_points"), "expected_points"),
        retrieved_citation_ids=_string_tuple(raw.get("retrieved_citation_ids"), "retrieved_citation_ids"),
        context=raw["context"],
        clean=_parse_answer(raw.get("clean"), "clean"),
        faulted=_parse_answer(raw.get("faulted"), "faulted"),
    )


def load_judge_set(path: Path) -> list[CalibrationPair]:
    """Parse `eval/calibration/judge-set.yaml`. A missing or empty file is zero pairs —
    the state of the repo before the first pair is authored, not an error."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise JudgeSetError(f"{path}: expected a YAML list of pairs, got {type(raw).__name__}")

    pairs: list[CalibrationPair] = []
    violations: list[str] = []
    for index, entry in enumerate(raw):
        ref = entry.get("golden_id") if isinstance(entry, Mapping) else None
        try:
            pairs.append(_parse_pair(entry))
        except ValueError as error:
            violations.append(f"{ref or f'index {index}'}: {error}")
    if violations:
        raise JudgeSetError(f"{len(violations)} judge-set schema violation(s):\n" + "\n".join(violations))
    return pairs


class BlockStyleDumper(yaml.SafeDumper):
    """Multi-line strings (the rendered context, long explanations) as `|` blocks, so the
    file diffs line by line like the golden set does."""


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.Node:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


BlockStyleDumper.add_representer(str, _represent_str)


def dump_judge_set(pairs: Sequence[CalibrationPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(
            [pair.as_dict() for pair in pairs],
            Dumper=BlockStyleDumper,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        ),
        encoding="utf-8",
    )


# --- validation ------------------------------------------------------------------------


def _cited(answer: CalibrationAnswer) -> set[str]:
    return {citation.article_id for citation in answer.envelope.fondement_juridique}


def pair_violations(pair: CalibrationPair) -> list[str]:
    """What makes a pair unable to test the fault it claims to plant. Structural checks
    only — whether the twin is a *good* fault is the author's call, not code's."""
    ref = pair.golden_id
    violations = []
    if pair.clean.envelope == pair.faulted.envelope:
        violations.append(f"{ref}: faulted twin is identical to its clean sibling")

    retrieved = set(pair.retrieved_citation_ids)
    archetype = pair.archetype
    if archetype is FaultArchetype.FABRICATED_CITATION:
        if not _cited(pair.faulted) - retrieved:
            violations.append(f"{ref}: fabricated_citation twin cites nothing outside retrieved_citation_ids")
        if _cited(pair.clean) - retrieved:
            violations.append(f"{ref}: clean answer already cites outside retrieved_citation_ids")
    if archetype in (FaultArchetype.DROPPED_EXPECTED_POINT, FaultArchetype.REFUSAL_WITHOUT_EXPLANATION) and not (
        pair.expected_points
    ):
        violations.append(f"{ref}: {archetype.value} needs expected_points to leave unasserted")
    if archetype is FaultArchetype.REFUSAL_WITHOUT_EXPLANATION:
        clean, faulted = pair.clean.envelope, pair.faulted.envelope
        if not (isinstance(clean, Refus) and isinstance(faulted, Refus)):
            violations.append(f"{ref}: refusal_without_explanation needs a Refus on both sides")
        else:
            if clean.motif != faulted.motif:
                violations.append(f"{ref}: refusal twin changes the motif — only the explaining half may go")
            if len(faulted.explanation) >= len(clean.explanation):
                violations.append(f"{ref}: refusal twin's explanation must be shorter than the clean one's")
    return violations


def validate_judge_set(pairs: Sequence[CalibrationPair], golden_set: Sequence[GoldenItem]) -> list[str]:
    """Every violation in `pairs`, checked against the golden set they are drawn from: one
    pair per golden item, and the pair's question/points are that item's own."""
    golden_by_id = {item.id: item for item in golden_set}
    violations = []
    for golden_id, count in Counter(pair.golden_id for pair in pairs).items():
        if count > 1:
            violations.append(f"{golden_id}: more than one pair for the same golden item")
    for pair in pairs:
        golden = golden_by_id.get(pair.golden_id)
        if golden is None:
            violations.append(f"{pair.golden_id}: not in the golden set")
        else:
            if pair.question != golden.question:
                violations.append(f"{pair.golden_id}: question differs from the golden item's")
            if pair.expected_points != golden.expected_points:
                violations.append(f"{pair.golden_id}: expected_points differ from the golden item's")
        violations.extend(pair_violations(pair))
    return violations


# --- the harness -----------------------------------------------------------------------


@dataclass(frozen=True)
class PairResult:
    golden_id: str
    archetype: FaultArchetype
    metric: JudgedMetric
    clean_score: float
    faulted_score: float
    detected: bool
    clean_error: ErrorDirection | None
    faulted_error: ErrorDirection | None
    clean_reasoning: str
    faulted_reasoning: str
    providers: tuple[str, ...]


def _judge_answer(judge: Judge, pair: CalibrationPair, answer: CalibrationAnswer, metric: JudgedMetric) -> JudgeScore:
    rendered = render_answer(answer.envelope)
    if metric is JudgedMetric.FAITHFULNESS:
        return judge.faithfulness(context=pair.context, answer=rendered)
    score = judge.point_coverage(question=pair.question, answer=rendered, expected_points=pair.expected_points)
    if score is None:
        raise JudgeSetError(f"{pair.golden_id}: point coverage needs expected_points")
    return score


def _error(score: float, label: HumanLabel, pass_threshold: float) -> ErrorDirection | None:
    judge_passes = score >= pass_threshold
    if judge_passes and label is HumanLabel.FAIL:
        return ErrorDirection.FALSE_PASS
    if not judge_passes and label is HumanLabel.PASS:
        return ErrorDirection.FALSE_FAIL
    return None


def score_pair(judge: Judge, pair: CalibrationPair, *, pass_threshold: float = DEFAULT_PASS_THRESHOLD) -> PairResult:
    """Two judge calls — the clean answer and its twin, on the archetype's target metric."""
    metric = TARGET_METRIC[pair.archetype]
    clean = _judge_answer(judge, pair, pair.clean, metric)
    faulted = _judge_answer(judge, pair, pair.faulted, metric)
    return PairResult(
        golden_id=pair.golden_id,
        archetype=pair.archetype,
        metric=metric,
        clean_score=clean.value,
        faulted_score=faulted.value,
        detected=faulted.value < clean.value,
        clean_error=_error(clean.value, pair.clean.human_label, pass_threshold),
        faulted_error=_error(faulted.value, pair.faulted.human_label, pass_threshold),
        clean_reasoning=clean.reasoning,
        faulted_reasoning=faulted.reasoning,
        providers=tuple(sorted({clean.provider, faulted.provider})),
    )


@dataclass(frozen=True)
class CalibrationSummary:
    pairs: int
    detected: int
    detection_required: int
    false_passes: int
    false_fails: int
    systematically_lenient: bool
    passed: bool


def summarize(results: Sequence[PairResult]) -> CalibrationSummary:
    """SPEC §12.10's two bars over `results`. "Systematic leniency" is read as *false
    passes outnumber false fails* — a judge whose only error is one false pass is already
    "a judge whose every error is a false pass"."""
    detected = sum(1 for result in results if result.detected)
    required = math.ceil(len(results) * DETECTION_BAR[0] / DETECTION_BAR[1])
    errors = [e for result in results for e in (result.clean_error, result.faulted_error) if e is not None]
    false_passes = errors.count(ErrorDirection.FALSE_PASS)
    false_fails = errors.count(ErrorDirection.FALSE_FAIL)
    lenient = false_passes > false_fails
    return CalibrationSummary(
        pairs=len(results),
        detected=detected,
        detection_required=required,
        false_passes=false_passes,
        false_fails=false_fails,
        systematically_lenient=lenient,
        passed=detected >= required and not lenient,
    )


@dataclass(frozen=True)
class CalibrationHeader:
    """Everything that could move a calibration verdict: the judge (model, resolved
    provider, prompt language per metric), the pass threshold, the set and the code."""

    run_id: str
    judge_model: str
    judge_provider: str
    judge_prompt_languages: dict[str, str]
    pass_threshold: float
    judge_set_git_sha: str
    code_git_sha: str
    timestamp: str


@dataclass(frozen=True)
class CalibrationRun:
    header: CalibrationHeader
    pairs: tuple[PairResult, ...]
    summary: CalibrationSummary


def run_calibration(
    pairs: Sequence[CalibrationPair],
    judge: Judge,
    *,
    run_id: str,
    repo_root: Path,
    judge_set_path: Path,
    runs_dir: Path,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> CalibrationRun:
    """Score every pair and persist the per-pair rows to `<runs_dir>/<run_id>.json` —
    per-pair, not just the verdict, the same reasoning SPEC §12.11 gives for per-item
    scores: "which pairs did it miss, and which way?" must stay answerable."""
    results = tuple(score_pair(judge, pair, pass_threshold=pass_threshold) for pair in pairs)
    header = CalibrationHeader(
        run_id=run_id,
        judge_model=judge.model,
        judge_provider=",".join(sorted({p for result in results for p in result.providers})),
        judge_prompt_languages=judge.languages(),
        pass_threshold=pass_threshold,
        judge_set_git_sha=resolve_git_sha(repo_root, path=judge_set_path),
        code_git_sha=resolve_git_sha(repo_root),
        timestamp=datetime.now(UTC).isoformat(),
    )
    run = CalibrationRun(header=header, pairs=results, summary=summarize(results))
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{run_id}.json").write_text(
        json.dumps(asdict(run), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return run


def load_calibration_run(path: Path) -> CalibrationRun:
    raw = json.loads(path.read_text(encoding="utf-8"))

    def pair_result(row: dict[str, Any]) -> PairResult:
        return PairResult(
            **{
                **row,
                "archetype": FaultArchetype(row["archetype"]),
                "metric": JudgedMetric(row["metric"]),
                "clean_error": ErrorDirection(row["clean_error"]) if row["clean_error"] else None,
                "faulted_error": ErrorDirection(row["faulted_error"]) if row["faulted_error"] else None,
                "providers": tuple(row["providers"]),
            }
        )

    return CalibrationRun(
        header=CalibrationHeader(**raw["header"]),
        pairs=tuple(pair_result(row) for row in raw["pairs"]),
        summary=CalibrationSummary(**raw["summary"]),
    )
