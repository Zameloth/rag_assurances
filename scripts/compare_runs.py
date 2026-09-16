#!/usr/bin/env python3
"""Run `compare.py`'s adoption rule over two committed run files (SPEC §12.7/§12.11, ADR-0011, #36).

**`compare.py` is the arbiter of the pre-registered rule — Langfuse is the trace viewer, not
the comparison surface** (SPEC §12.11). This script never touches Langfuse or Qdrant: it
reads only the two `eval/runs/<run-id>.json` files named on the command line and applies
`rag.eval.compare`'s paired-delta math to them.

    uv run python scripts/compare_runs.py \\
        eval/runs/rung3-incumbent.json eval/runs/rung3-challenger.json article_recall_at_4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.eval.compare import (
    DECISION_METRICS,
    DISCORDANT_ADOPTION_THRESHOLD,
    ComparisonReport,
    compare_run_files,
    write_comparison,
)


def _print_report(report: ComparisonReport) -> None:
    print(f"incumbent : {report.incumbent_run_id}")
    print(f"challenger: {report.challenger_run_id}")
    print(f"primary   : {report.primary_metric}\n")

    for metric, comparison in report.metrics.items():
        marker = "*" if metric in DECISION_METRICS else " "
        print(
            f"{marker} {metric:<28} n={comparison.n_paired:>2}  "
            f"improved={comparison.improved:>2}  regressed={comparison.regressed:>2}  "
            f"net={comparison.net_discordant:+d}  sign-test p={comparison.sign_test_p:.3f}"
        )
        if comparison.regressed_items:
            print(f"    regressed: {', '.join(comparison.regressed_items)}")

    verdict = report.verdict
    print()
    if verdict.adopt:
        print(f"VERDICT: ADOPT — net discordant on {verdict.primary_metric} = {verdict.net_discordant_primary:+d}")
        return

    reasons = []
    if not verdict.threshold_met:
        reasons.append(
            f"primary net discordant {verdict.net_discordant_primary:+d} "
            f"< {DISCORDANT_ADOPTION_THRESHOLD}"
        )
    if verdict.guard_violations:
        reasons.append(f"guard violated on {', '.join(verdict.guard_violations)}")
    print(f"VERDICT: KEEP INCUMBENT (inconclusive resolves to no change) — {'; '.join(reasons)}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("incumbent", type=Path, help="eval/runs/<run-id>.json for the incumbent arm")
    parser.add_argument("challenger", type=Path, help="eval/runs/<run-id>.json for the challenger arm")
    parser.add_argument("primary_metric", choices=DECISION_METRICS, help="this rung's pre-registered primary")
    parser.add_argument(
        "--out", type=Path, default=None, help="also persist the verdict as JSON here (SPEC §12.11)"
    )
    args = parser.parse_args(argv)

    report = compare_run_files(args.incumbent, args.challenger, args.primary_metric)
    _print_report(report)
    if args.out is not None:
        write_comparison(report, args.out)
        print(f"\nverdict written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
