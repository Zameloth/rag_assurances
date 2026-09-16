"""`compare.py` — the paired-delta arbiter of ADR-0011's adoption rule, SPEC §12.7/§12.11, #36."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.eval.compare import (
    DECISION_METRICS,
    DISCORDANT_ADOPTION_THRESHOLD,
    GUARD_MAX_NET_REGRESSION,
    compare_run_files,
    compare_runs,
    metric_names,
    report_to_dict,
    write_comparison,
)
from rag.eval.retrieval_metrics import ItemRetrievalScore
from rag.eval.retrieval_run import RetrievalRun, RunHeader, write_run


def header(run_id: str) -> RunHeader:
    return RunHeader(
        run_id=run_id,
        rung="rung3",
        arm=run_id,
        golden_set_git_sha="a" * 40,
        langfuse_dataset_version="2026-09-16T00:00:00Z",
        retrieval_config={},
        code_git_sha="b" * 40,
        timestamp="2026-09-16T12:00:00Z",
        langfuse_run_name=run_id,
    )


def score(item_id: str, **overrides: object) -> ItemRetrievalScore:
    defaults: dict[str, object] = dict(
        item_id=item_id,
        short_circuit_path="no_reference",
        fiche_recall_at_4=None,
        fiche_recall_at_10=None,
        fiche_recall_at_candidate=None,
        article_recall_at_4=None,
        article_recall_at_10=None,
        article_recall_at_candidate=None,
        zero_articles=None,
        floor_correct=None,
        span_containment_at_4=None,
    )
    defaults.update(overrides)
    return ItemRetrievalScore(**defaults)  # type: ignore[arg-type]


def run(run_id: str, items: tuple[ItemRetrievalScore, ...]) -> RetrievalRun:
    return RetrievalRun(header=header(run_id), items=items)


class TestMetricNames:
    def test_lists_every_metric_but_id_and_short_circuit_path(self) -> None:
        names = metric_names()
        assert "item_id" not in names
        assert "short_circuit_path" not in names
        assert set(names) == {
            "fiche_recall_at_4",
            "fiche_recall_at_10",
            "fiche_recall_at_candidate",
            "article_recall_at_4",
            "article_recall_at_10",
            "article_recall_at_candidate",
            "zero_articles",
            "floor_correct",
            "span_containment_at_4",
        }

    def test_decision_metrics_are_exactly_the_adr_0011_four(self) -> None:
        assert set(DECISION_METRICS) == {
            "fiche_recall_at_4",
            "article_recall_at_4",
            "zero_articles",
            "floor_correct",
        }
        assert set(DECISION_METRICS) <= set(metric_names())


class TestPerItemDirection:
    """Higher/`True` is better on every metric except `zero_articles`, where `True` is the
    failure mode ADR-0011 introduced the metric to catch."""

    def test_higher_recall_is_improved(self) -> None:
        incumbent = run("incumbent", (score("gs-001", fiche_recall_at_4=0.5),))
        challenger = run("challenger", (score("gs-001", fiche_recall_at_4=1.0),))
        report = compare_runs(incumbent, challenger, "fiche_recall_at_4")
        comparison = report.metrics["fiche_recall_at_4"]
        assert comparison.improved == 1
        assert comparison.regressed == 0
        assert comparison.regressed_items == ()
        assert comparison.improved_items == ("gs-001",)

    def test_lower_recall_is_regressed(self) -> None:
        incumbent = run("incumbent", (score("gs-001", article_recall_at_4=1.0),))
        challenger = run("challenger", (score("gs-001", article_recall_at_4=0.5),))
        comparison = compare_runs(incumbent, challenger, "article_recall_at_4").metrics["article_recall_at_4"]
        assert comparison.regressed == 1
        assert comparison.regressed_items == ("gs-001",)

    def test_zero_articles_true_to_false_is_improved(self) -> None:
        """Incumbent returned no articles at all (the failure mode); the challenger fixed it."""
        incumbent = run("incumbent", (score("gs-001", zero_articles=True),))
        challenger = run("challenger", (score("gs-001", zero_articles=False),))
        comparison = compare_runs(incumbent, challenger, "zero_articles").metrics["zero_articles"]
        assert comparison.improved == 1
        assert comparison.regressed == 0

    def test_zero_articles_false_to_true_is_regressed(self) -> None:
        """The challenger newly fails to return any article — a regression despite `True` reading higher."""
        incumbent = run("incumbent", (score("gs-001", zero_articles=False),))
        challenger = run("challenger", (score("gs-001", zero_articles=True),))
        comparison = compare_runs(incumbent, challenger, "zero_articles").metrics["zero_articles"]
        assert comparison.regressed == 1
        assert comparison.improved == 0

    def test_floor_correct_false_to_true_is_improved(self) -> None:
        incumbent = run("incumbent", (score("gs-001", floor_correct=False),))
        challenger = run("challenger", (score("gs-001", floor_correct=True),))
        comparison = compare_runs(incumbent, challenger, "floor_correct").metrics["floor_correct"]
        assert comparison.improved == 1

    def test_equal_values_are_tied_not_discordant(self) -> None:
        incumbent = run("incumbent", (score("gs-001", fiche_recall_at_4=0.75),))
        challenger = run("challenger", (score("gs-001", fiche_recall_at_4=0.75),))
        comparison = compare_runs(incumbent, challenger, "fiche_recall_at_4").metrics["fiche_recall_at_4"]
        assert comparison.improved == 0
        assert comparison.regressed == 0
        assert comparison.n_paired == 1
        assert comparison.deltas[0].outcome == "tied"

    def test_none_on_either_side_is_excluded_from_the_pair(self) -> None:
        """`reponse_sans_article` items carry `article_recall_at_4=None` on both runs (SPEC
        §12.6: undefined, not zero) — the pair contributes nothing to this metric's count."""
        incumbent = run("incumbent", (score("gs-001", article_recall_at_4=None),))
        challenger = run("challenger", (score("gs-001", article_recall_at_4=0.5),))
        comparison = compare_runs(incumbent, challenger, "article_recall_at_4").metrics["article_recall_at_4"]
        assert comparison.n_paired == 0
        assert comparison.improved == 0
        assert comparison.regressed == 0
        assert comparison.deltas[0].outcome is None


class TestSignTestP:
    """Cross-checked against ADR-0011's own worked example: "clearing p<0.05 two-sided needs
    roughly 6 discordant pairs all one-way"."""

    def _items(self, n_improved: int, n_regressed: int) -> tuple[RetrievalRun, RetrievalRun]:
        incumbent_items = []
        challenger_items = []
        for i in range(n_improved):
            incumbent_items.append(score(f"gs-imp-{i}", fiche_recall_at_4=0.0))
            challenger_items.append(score(f"gs-imp-{i}", fiche_recall_at_4=1.0))
        for i in range(n_regressed):
            incumbent_items.append(score(f"gs-reg-{i}", fiche_recall_at_4=1.0))
            challenger_items.append(score(f"gs-reg-{i}", fiche_recall_at_4=0.0))
        return run("incumbent", tuple(incumbent_items)), run("challenger", tuple(challenger_items))

    def test_six_all_one_way_clears_p_below_point_zero_five(self) -> None:
        incumbent, challenger = self._items(6, 0)
        comparison = compare_runs(incumbent, challenger, "fiche_recall_at_4").metrics["fiche_recall_at_4"]
        assert comparison.sign_test_p < 0.05

    def test_five_one_way_one_against_does_not_clear_it(self) -> None:
        incumbent, challenger = self._items(5, 1)
        comparison = compare_runs(incumbent, challenger, "fiche_recall_at_4").metrics["fiche_recall_at_4"]
        assert comparison.sign_test_p >= 0.05

    def test_no_discordant_pairs_is_p_one(self) -> None:
        incumbent = run("incumbent", (score("gs-001", fiche_recall_at_4=0.5),))
        challenger = run("challenger", (score("gs-001", fiche_recall_at_4=0.5),))
        comparison = compare_runs(incumbent, challenger, "fiche_recall_at_4").metrics["fiche_recall_at_4"]
        assert comparison.sign_test_p == 1.0

    def test_is_symmetric_in_which_side_is_the_minority(self) -> None:
        a, b = self._items(4, 1)
        c, d = self._items(1, 4)
        p1 = compare_runs(a, b, "fiche_recall_at_4").metrics["fiche_recall_at_4"].sign_test_p
        p2 = compare_runs(c, d, "fiche_recall_at_4").metrics["fiche_recall_at_4"].sign_test_p
        assert p1 == pytest.approx(p2)


class TestVerdict:
    def _make(
        self, *, primary_net: int, other_nets: dict[str, int] | None = None
    ) -> tuple[RetrievalRun, RetrievalRun]:
        """Builds a pair of runs whose `article_recall_at_4` net discordant is `primary_net`
        (all in one direction, so it's also the improved-regressed count) and whose other
        decision metrics carry whatever net discordant `other_nets` asks for (defaulting to
        0, i.e. no guard pressure)."""
        other_nets = other_nets or {}
        incumbent_items = []
        challenger_items = []
        idx = 0
        for _ in range(abs(primary_net)):
            incumbent_val, challenger_val = (0.0, 1.0) if primary_net >= 0 else (1.0, 0.0)
            incumbent_items.append(score(f"gs-{idx}", article_recall_at_4=incumbent_val))
            challenger_items.append(score(f"gs-{idx}", article_recall_at_4=challenger_val))
            idx += 1
        for metric, net in other_nets.items():
            for _ in range(abs(net)):
                bad_first = net < 0
                if metric == "zero_articles":
                    incumbent_val = not bad_first
                    challenger_val = bool(bad_first)
                else:
                    incumbent_val = 0.0 if not bad_first else 1.0
                    challenger_val = 1.0 if not bad_first else 0.0
                incumbent_items.append(score(f"gs-{idx}", **{metric: incumbent_val}))
                challenger_items.append(score(f"gs-{idx}", **{metric: challenger_val}))
                idx += 1
        return run("incumbent", tuple(incumbent_items)), run("challenger", tuple(challenger_items))

    def test_adopts_when_threshold_met_and_no_guard_violation(self) -> None:
        incumbent, challenger = self._make(primary_net=DISCORDANT_ADOPTION_THRESHOLD)
        verdict = compare_runs(incumbent, challenger, "article_recall_at_4").verdict
        assert verdict.threshold_met is True
        assert verdict.guard_violations == ()
        assert verdict.adopt is True

    def test_keeps_incumbent_below_threshold(self) -> None:
        incumbent, challenger = self._make(primary_net=DISCORDANT_ADOPTION_THRESHOLD - 1)
        verdict = compare_runs(incumbent, challenger, "article_recall_at_4").verdict
        assert verdict.threshold_met is False
        assert verdict.adopt is False

    def test_negative_primary_net_keeps_incumbent(self) -> None:
        incumbent, challenger = self._make(primary_net=-DISCORDANT_ADOPTION_THRESHOLD)
        verdict = compare_runs(incumbent, challenger, "article_recall_at_4").verdict
        assert verdict.net_discordant_primary == -DISCORDANT_ADOPTION_THRESHOLD
        assert verdict.threshold_met is False
        assert verdict.adopt is False

    def test_guard_blocks_adoption_even_when_primary_clears(self) -> None:
        incumbent, challenger = self._make(
            primary_net=DISCORDANT_ADOPTION_THRESHOLD,
            other_nets={"fiche_recall_at_4": -(GUARD_MAX_NET_REGRESSION + 1)},
        )
        verdict = compare_runs(incumbent, challenger, "article_recall_at_4").verdict
        assert verdict.threshold_met is True
        assert verdict.guard_violations == ("fiche_recall_at_4",)
        assert verdict.adopt is False

    def test_guard_allows_exactly_the_boundary_regression(self) -> None:
        incumbent, challenger = self._make(
            primary_net=DISCORDANT_ADOPTION_THRESHOLD,
            other_nets={"floor_correct": -GUARD_MAX_NET_REGRESSION},
        )
        verdict = compare_runs(incumbent, challenger, "article_recall_at_4").verdict
        assert verdict.guard_violations == ()
        assert verdict.adopt is True

    def test_guard_never_checks_the_primary_against_itself(self) -> None:
        """A rung whose primary itself is `article_recall_at_4` regressing hard is caught by
        the threshold, not double-counted as its own guard violation."""
        incumbent, challenger = self._make(primary_net=-10)
        verdict = compare_runs(incumbent, challenger, "article_recall_at_4").verdict
        assert "article_recall_at_4" not in verdict.guard_violations

    def test_zero_articles_guard_direction_matches_its_inverted_metric(self) -> None:
        """A challenger that newly fails on `zero_articles` (net discordant negative in
        improvement terms) must trip the guard exactly like any other decision metric."""
        incumbent, challenger = self._make(
            primary_net=DISCORDANT_ADOPTION_THRESHOLD,
            other_nets={"zero_articles": -(GUARD_MAX_NET_REGRESSION + 1)},
        )
        verdict = compare_runs(incumbent, challenger, "article_recall_at_4").verdict
        assert verdict.guard_violations == ("zero_articles",)
        assert verdict.adopt is False

    def test_rejects_a_non_decision_primary(self) -> None:
        incumbent = run("incumbent", (score("gs-001", span_containment_at_4=0.5),))
        challenger = run("challenger", (score("gs-001", span_containment_at_4=1.0),))
        with pytest.raises(ValueError):
            compare_runs(incumbent, challenger, "span_containment_at_4")


class TestPairing:
    def test_mismatched_item_ids_raise(self) -> None:
        incumbent = run("incumbent", (score("gs-001"),))
        challenger = run("challenger", (score("gs-002"),))
        with pytest.raises(ValueError):
            compare_runs(incumbent, challenger, "fiche_recall_at_4")

    def test_deltas_are_reported_in_incumbent_item_order(self) -> None:
        incumbent = run(
            "incumbent",
            (score("gs-003", fiche_recall_at_4=0.0), score("gs-001", fiche_recall_at_4=0.0)),
        )
        challenger = run(
            "challenger",
            (score("gs-001", fiche_recall_at_4=1.0), score("gs-003", fiche_recall_at_4=1.0)),
        )
        comparison = compare_runs(incumbent, challenger, "fiche_recall_at_4").metrics["fiche_recall_at_4"]
        assert [d.item_id for d in comparison.deltas] == ["gs-003", "gs-001"]

    def test_report_names_both_run_ids(self) -> None:
        incumbent = run("incumbent-run", (score("gs-001", fiche_recall_at_4=0.5),))
        challenger = run("challenger-run", (score("gs-001", fiche_recall_at_4=0.5),))
        report = compare_runs(incumbent, challenger, "fiche_recall_at_4")
        assert report.incumbent_run_id == "incumbent-run"
        assert report.challenger_run_id == "challenger-run"


class TestCompareRunFiles:
    def test_reads_two_persisted_run_files(self, tmp_path: Path) -> None:
        incumbent = run("incumbent", (score("gs-001", article_recall_at_4=0.0),))
        challenger = run("challenger", (score("gs-001", article_recall_at_4=1.0),))
        incumbent_path = write_run(incumbent, tmp_path)
        challenger_path = write_run(challenger, tmp_path)

        report = compare_run_files(incumbent_path, challenger_path, "article_recall_at_4")
        assert report.incumbent_run_id == "incumbent"
        assert report.challenger_run_id == "challenger"
        assert report.metrics["article_recall_at_4"].improved == 1


class TestPersistence:
    def test_report_to_dict_carries_the_verdict_and_sign_test_p(self) -> None:
        incumbent, challenger = (
            run("incumbent", (score("gs-001", article_recall_at_4=0.0),)),
            run("challenger", (score("gs-001", article_recall_at_4=1.0),)),
        )
        report = compare_runs(incumbent, challenger, "article_recall_at_4")
        payload = report_to_dict(report)
        assert payload["verdict"]["adopt"] is False  # net discordant is only 1, below threshold
        assert payload["metrics"]["article_recall_at_4"]["sign_test_p"] == pytest.approx(1.0)
        assert payload["metrics"]["article_recall_at_4"]["improved_items"] == ["gs-001"]
        assert payload["metrics"]["article_recall_at_4"]["is_decision_metric"] is True
        assert payload["metrics"]["span_containment_at_4"]["is_decision_metric"] is False

    def test_write_comparison_persists_valid_json(self, tmp_path: Path) -> None:
        incumbent, challenger = self._four_item_win()
        report = compare_runs(incumbent, challenger, "article_recall_at_4")

        out_path = tmp_path / "nested" / "verdict.json"
        result = write_comparison(report, out_path)

        assert result == out_path
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["verdict"]["adopt"] is True
        assert payload["primary_metric"] == "article_recall_at_4"

    @staticmethod
    def _four_item_win() -> tuple[RetrievalRun, RetrievalRun]:
        incumbent_items = tuple(score(f"gs-{i}", article_recall_at_4=0.0) for i in range(4))
        challenger_items = tuple(score(f"gs-{i}", article_recall_at_4=1.0) for i in range(4))
        return run("incumbent", incumbent_items), run("challenger", challenger_items)
