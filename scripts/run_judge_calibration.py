#!/usr/bin/env python3
"""Run the judge over the calibration set and report whether it can detect a fault
(SPEC §12.10, ADR-0025, #47).

Scores every pair in `eval/calibration/judge-set.yaml` on the judged metric its archetype
targets — two judge calls per pair — and prints, **per pair**, whether the faulted twin
scored strictly lower than its clean sibling and the direction of every error (false pass /
false fail, against the human labels). Then the two bars: detection on ≥ 10 of 12 pairs,
and no systematic leniency. Per-pair rows and the verdict are written to
`eval/runs/calibration-<stamp>.json`.

**This is also the FR/EN experiment and the judge regression test.** Prompt language is
config (`JUDGE_FAITHFULNESS_LANGUAGE`, `JUDGE_POINT_COVERAGE_LANGUAGE`), so the other
configuration is one environment variable away:

    uv run python scripts/run_judge_calibration.py
    JUDGE_FAITHFULNESS_LANGUAGE=fr uv run python scripts/run_judge_calibration.py

Re-run it whenever the judge model, a judge prompt, or OpenRouter's resolved provider moves.
Costs real OpenRouter calls (~2 per pair) but no Qdrant and no Langfuse.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.config import load_settings
from rag.eval.calibration import (
    DEFAULT_PASS_THRESHOLD,
    CalibrationRun,
    load_judge_set,
    run_calibration,
    validate_judge_set,
)
from rag.eval.judge_chain import make_judge
from rag.eval.schema import load_golden_set

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGE_SET_PATH = REPO_ROOT / "eval" / "calibration" / "judge-set.yaml"
GOLDEN_SET_PATH = REPO_ROOT / "eval" / "golden" / "golden-set.yaml"
RUNS_DIR = REPO_ROOT / "eval" / "runs"


def _print_report(run: CalibrationRun) -> None:
    header = run.header
    print(f"judge: {header.judge_model} via {header.judge_provider} — prompts {header.judge_prompt_languages}")
    print(f"pass threshold: {header.pass_threshold}")
    print()
    for pair in run.pairs:
        mark = "detected" if pair.detected else "MISSED  "
        errors = ", ".join(
            f"{side} {error.value}"
            for side, error in (("clean", pair.clean_error), ("faulted", pair.faulted_error))
            if error is not None
        )
        print(
            f"  {pair.golden_id}  {mark}  {pair.archetype.value:<28} {pair.metric.value:<15}"
            f" clean {pair.clean_score:.2f}  faulted {pair.faulted_score:.2f}"
            + (f"  [{errors}]" if errors else "")
        )
    summary = run.summary
    print()
    print(f"detection: {summary.detected}/{summary.pairs} (bar: {summary.detection_required})")
    print(f"errors: {summary.false_passes} false pass(es), {summary.false_fails} false fail(s)")
    if summary.systematically_lenient:
        print("systematic leniency: false passes outnumber false fails — disqualifying (SPEC §12.10)")
    print(f"verdict: {'PASS' if summary.passed else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=DEFAULT_PASS_THRESHOLD,
        help=f"score at or above which the judge is read as passing an answer (default {DEFAULT_PASS_THRESHOLD})",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    pairs = load_judge_set(JUDGE_SET_PATH)
    violations = validate_judge_set(pairs, load_golden_set(GOLDEN_SET_PATH))
    if violations:
        print(f"{JUDGE_SET_PATH}: INVALID\n" + "\n".join(violations), file=sys.stderr)
        return 1
    if not pairs:
        print(f"{JUDGE_SET_PATH}: no pairs yet — author some with scripts/author_calibration_pair.py", file=sys.stderr)
        return 1

    run_id = f"calibration-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run = run_calibration(
        pairs,
        make_judge(settings),
        run_id=run_id,
        repo_root=REPO_ROOT,
        judge_set_path=JUDGE_SET_PATH,
        runs_dir=RUNS_DIR,
        pass_threshold=args.pass_threshold,
    )
    print(f"written to eval/runs/{run_id}.json")
    _print_report(run)
    return 0 if run.summary.passed else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
