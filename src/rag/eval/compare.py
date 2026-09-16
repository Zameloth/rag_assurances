"""The paired-delta arbiter of the pre-registered adoption rule — SPEC §12.7/§12.11, ADR-0011, #36.

**Per-item paired deltas, never two independent proportions.** Every rung runs the same
golden-set items, so "did the challenger do better" is answered item by item against the
incumbent's own score on that item, not by comparing two means. That is what buys the
sensitivity ADR-0011 needs at N=36-40: a metric that moves on only a handful of items still
shows up as a net discordant count, where two independent proportions would wash it out.

**Higher/`True` is better on every metric except `zero_articles`.** SPEC §12.6 introduced
zero-article rate as a decision metric precisely because recall cannot be punished for
over-retrieval — `zero_articles=True` records the failure the metric exists to catch, so a
challenger turning it `False` is the improvement, not the reading `True > False` would
suggest.

**Undefined-on-either-side pairs are excluded, not zeroed.** `ItemRetrievalScore` already
reads `None` wherever a metric does not apply to an item (SPEC §12.6); a pair where either
run has `None` contributes nothing to that metric's discordant count, the same "undefined,
not zero" posture `rag.eval.retrieval_metrics` keeps.

**`compare.py` is the arbiter — Langfuse is the trace viewer, not the comparison surface**
(SPEC §12.11): this module reads only the two persisted `eval/runs/<run-id>.json` files,
never a Langfuse dataset or trace.

**The verdict itself is persisted, not just printed** — SPEC §12.11's "git holds the
decision" and #36's "sign-test p is computed and persisted alongside" both read as more than
a stdout report: `write_comparison` writes the same JSON shape `rag.eval.retrieval_run`
writes for a single run, so a rung's adoption call survives past its Langfuse traces'
30-day retention exactly like the per-item scores it was computed from.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

from rag.eval.retrieval_metrics import ItemRetrievalScore
from rag.eval.retrieval_run import RetrievalRun, load_run

__all__ = [
    "DECISION_METRICS",
    "DISCORDANT_ADOPTION_THRESHOLD",
    "GUARD_MAX_NET_REGRESSION",
    "ComparisonReport",
    "ItemDelta",
    "MetricComparison",
    "Verdict",
    "compare_run_files",
    "compare_runs",
    "metric_names",
    "report_to_dict",
    "write_comparison",
]

Outcome = Literal["improved", "regressed", "tied"]

_NON_METRIC_FIELDS = frozenset({"item_id", "short_circuit_path"})

# SPEC §12.6 / ADR-0011: "zero-article rate" is the one metric where `True` is the failure
# mode. Every other metric — the recall family, floor correctness, span containment — reads
# "higher/`True` is better" on its face.
_INVERTED_METRICS = frozenset({"zero_articles"})

# SPEC §12.6 / ADR-0011's table: the four metrics a rung's primary must be drawn from.
DECISION_METRICS = ("fiche_recall_at_4", "article_recall_at_4", "zero_articles", "floor_correct")

# SPEC §12.7 rule 2 — "net discordant pairs on the primary ≥ 4 items (≈11pp at N=36)".
DISCORDANT_ADOPTION_THRESHOLD = 4

# SPEC §12.7 rule 2 — "no other decision metric regresses by more than 1 net item".
GUARD_MAX_NET_REGRESSION = 1


def metric_names() -> tuple[str, ...]:
    """Every scored field of `ItemRetrievalScore` — the nine numbers SPEC §12.6 names — read
    off the dataclass itself rather than duplicated as a literal tuple, so a future field
    added there is compared here without a second edit."""
    return tuple(f.name for f in fields(ItemRetrievalScore) if f.name not in _NON_METRIC_FIELDS)


@dataclass(frozen=True)
class ItemDelta:
    """One golden-set item's paired reading on one metric. `outcome` is `None` exactly when
    the metric is undefined (SPEC §12.6's `None`) on either run — the pair still appears
    here, with whichever side has a value, so "which items regressed" stays answerable
    without hiding items a metric never applied to."""

    item_id: str
    incumbent: float | bool | None
    challenger: float | bool | None
    outcome: Outcome | None


@dataclass(frozen=True)
class MetricComparison:
    """One metric's discordant-pair count across a run pair, plus the diagnostic sign-test
    *p* — ADR-0011: persisted for every metric, but never the gate."""

    metric: str
    deltas: tuple[ItemDelta, ...]

    @property
    def n_paired(self) -> int:
        """Items where this metric is defined on both runs — includes ties, since a tie is
        still a comparable pair, just not a discordant one."""
        return sum(1 for delta in self.deltas if delta.outcome is not None)

    @property
    def improved(self) -> int:
        return sum(1 for delta in self.deltas if delta.outcome == "improved")

    @property
    def regressed(self) -> int:
        return sum(1 for delta in self.deltas if delta.outcome == "regressed")

    @property
    def net_discordant(self) -> int:
        """Positive favours the challenger — SPEC §12.7's "net discordant pairs" is exactly
        this signed count, not the raw discordant total."""
        return self.improved - self.regressed

    @property
    def sign_test_p(self) -> float:
        return _sign_test_p(self.improved, self.regressed)

    @property
    def improved_items(self) -> tuple[str, ...]:
        return tuple(delta.item_id for delta in self.deltas if delta.outcome == "improved")

    @property
    def regressed_items(self) -> tuple[str, ...]:
        return tuple(delta.item_id for delta in self.deltas if delta.outcome == "regressed")


@dataclass(frozen=True)
class Verdict:
    """The pre-registered rule's output — SPEC §12.7 rules 2-3: adopt, or keep the incumbent
    with inconclusive resolving to *no change*, never to a judgement call."""

    primary_metric: str
    net_discordant_primary: int
    threshold_met: bool
    guard_violations: tuple[str, ...]
    adopt: bool


@dataclass(frozen=True)
class ComparisonReport:
    incumbent_run_id: str
    challenger_run_id: str
    primary_metric: str
    metrics: dict[str, MetricComparison]
    verdict: Verdict


def _sign_test_p(improved: int, regressed: int) -> float:
    """Two-sided exact binomial sign-test *p* on the discordant pairs (`improved` + `regressed`
    trials, split evenly under the null). Cross-checked against ADR-0011's own worked example:
    "clearing p<0.05 two-sided needs roughly 6 discordant pairs all one-way" — `n=6, k=0` here
    gives `p ≈ 0.031`."""
    n = improved + regressed
    if n == 0:
        return 1.0
    k = min(improved, regressed)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _outcome(metric: str, incumbent_value: object, challenger_value: object) -> Outcome | None:
    if incumbent_value is None or challenger_value is None:
        return None
    incumbent_num = float(incumbent_value)  # type: ignore[arg-type]
    challenger_num = float(challenger_value)  # type: ignore[arg-type]
    if incumbent_num == challenger_num:
        return "tied"
    challenger_reads_higher = challenger_num > incumbent_num
    challenger_is_better = not challenger_reads_higher if metric in _INVERTED_METRICS else challenger_reads_higher
    return "improved" if challenger_is_better else "regressed"


def compare_runs(incumbent: RetrievalRun, challenger: RetrievalRun, primary_metric: str) -> ComparisonReport:
    """Pair `incumbent` and `challenger` by item id and score every metric SPEC §12.6 names,
    then apply SPEC §12.7's adoption rule against `primary_metric`.

    Raises `ValueError` if `primary_metric` is not one of `DECISION_METRICS` — the pre-
    registered table (SPEC §12.7) only ever names a primary from that set — or if the two
    runs disagree on which items they scored, since a paired comparison has nothing to pair
    against an item the other run never ran.
    """
    if primary_metric not in DECISION_METRICS:
        raise ValueError(f"primary metric must be one of {DECISION_METRICS}, got {primary_metric!r}")

    incumbent_by_id = {item.item_id: item for item in incumbent.items}
    challenger_by_id = {item.item_id: item for item in challenger.items}
    if incumbent_by_id.keys() != challenger_by_id.keys():
        only_incumbent = sorted(incumbent_by_id.keys() - challenger_by_id.keys())
        only_challenger = sorted(challenger_by_id.keys() - incumbent_by_id.keys())
        raise ValueError(
            "runs are not paired — same golden-set item ids are required on both sides "
            f"(only in incumbent: {only_incumbent}, only in challenger: {only_challenger})"
        )

    item_ids = [item.item_id for item in incumbent.items]
    metrics: dict[str, MetricComparison] = {}
    for metric in metric_names():
        deltas = []
        for item_id in item_ids:
            incumbent_value = getattr(incumbent_by_id[item_id], metric)
            challenger_value = getattr(challenger_by_id[item_id], metric)
            deltas.append(
                ItemDelta(
                    item_id=item_id,
                    incumbent=incumbent_value,
                    challenger=challenger_value,
                    outcome=_outcome(metric, incumbent_value, challenger_value),
                )
            )
        metrics[metric] = MetricComparison(metric=metric, deltas=tuple(deltas))

    primary_comparison = metrics[primary_metric]
    threshold_met = primary_comparison.net_discordant >= DISCORDANT_ADOPTION_THRESHOLD
    guard_violations = tuple(
        metric
        for metric in DECISION_METRICS
        if metric != primary_metric and -metrics[metric].net_discordant > GUARD_MAX_NET_REGRESSION
    )
    verdict = Verdict(
        primary_metric=primary_metric,
        net_discordant_primary=primary_comparison.net_discordant,
        threshold_met=threshold_met,
        guard_violations=guard_violations,
        adopt=threshold_met and not guard_violations,
    )
    return ComparisonReport(
        incumbent_run_id=incumbent.header.run_id,
        challenger_run_id=challenger.header.run_id,
        primary_metric=primary_metric,
        metrics=metrics,
        verdict=verdict,
    )


def compare_run_files(incumbent_path: Path, challenger_path: Path, primary_metric: str) -> ComparisonReport:
    """`compare_runs` from the two persisted `eval/runs/<run-id>.json` files SPEC §12.11
    writes, rather than already-loaded `RetrievalRun`s — the seam the CLI runs through."""
    return compare_runs(load_run(incumbent_path), load_run(challenger_path), primary_metric)


def report_to_dict(report: ComparisonReport) -> dict[str, Any]:
    """`report`'s JSON-shaped view — the computed properties (`net_discordant`, `sign_test_p`,
    `regressed_items`, ...) alongside the raw per-item deltas, since `dataclasses.asdict`
    alone would only reach `MetricComparison`'s two stored fields and drop everything derived
    from them."""
    return {
        "incumbent_run_id": report.incumbent_run_id,
        "challenger_run_id": report.challenger_run_id,
        "primary_metric": report.primary_metric,
        "metrics": {
            metric: {
                "n_paired": comparison.n_paired,
                "improved": comparison.improved,
                "regressed": comparison.regressed,
                "net_discordant": comparison.net_discordant,
                "sign_test_p": comparison.sign_test_p,
                "improved_items": list(comparison.improved_items),
                "regressed_items": list(comparison.regressed_items),
                "is_decision_metric": metric in DECISION_METRICS,
            }
            for metric, comparison in report.metrics.items()
        },
        "verdict": {
            "primary_metric": report.verdict.primary_metric,
            "net_discordant_primary": report.verdict.net_discordant_primary,
            "threshold_met": report.verdict.threshold_met,
            "guard_violations": list(report.verdict.guard_violations),
            "adopt": report.verdict.adopt,
        },
    }


def write_comparison(report: ComparisonReport, path: Path) -> Path:
    """Persist `report` as JSON at `path` — SPEC §12.11's "git holds the decision": the
    verdict and the diagnostic sign-test *p* survive here past the 30-day window Langfuse
    traces do not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
