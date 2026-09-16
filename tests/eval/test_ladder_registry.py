"""The pre-registered ladder table — SPEC §12.7/§12.8, ADR-0011, #37.

Committed once, before a single rung or pre-ladder A/B runs — this is the ordering
constraint ADR-0011 calls the one thing that cannot be recovered afterwards. These tests
pin the table to the exact SPEC §12.7/§12.8 values, since a typo here is a silently wrong
primary metric for a real rung.
"""

from __future__ import annotations

import pytest

from rag.eval.compare import DECISION_METRICS
from rag.eval.ladder_registry import LADDER_TABLE, primary_metric_for


class TestLadderTable:
    def test_has_exactly_the_six_rungs_and_two_pre_ladder_abs(self) -> None:
        rungs = {spec.rung for spec in LADDER_TABLE}
        assert rungs == {
            "rung1",
            "rung2",
            "rung3",
            "rung4",
            "rung5",
            "rung6",
            "ab_article_breadcrumb",
            "ab_fiche_header",
        }

    def test_rung_one_has_no_primary_reference_floor_not_a_comparison(self) -> None:
        spec = next(spec for spec in LADDER_TABLE if spec.rung == "rung1")
        assert spec.primary is None

    @pytest.mark.parametrize(
        ("rung", "expected_primary"),
        [
            ("rung2", "article_recall_at_4"),
            ("rung3", "article_recall_at_4"),
            ("rung4", "fiche_recall_at_4"),
            ("rung5", "zero_articles"),
            ("rung6", "fiche_recall_at_4"),
            ("ab_article_breadcrumb", "article_recall_at_4"),
            ("ab_fiche_header", "fiche_recall_at_4"),
        ],
    )
    def test_matches_the_spec_table(self, rung: str, expected_primary: str) -> None:
        assert primary_metric_for(rung) == expected_primary

    def test_every_registered_primary_is_a_decision_metric(self) -> None:
        for spec in LADDER_TABLE:
            if spec.primary is not None:
                assert spec.primary in DECISION_METRICS

    def test_rung_ids_are_unique(self) -> None:
        rungs = [spec.rung for spec in LADDER_TABLE]
        assert len(rungs) == len(set(rungs))


class TestPrimaryMetricFor:
    def test_raises_for_rung_one_which_has_no_pre_registered_primary(self) -> None:
        with pytest.raises(ValueError, match="rung1"):
            primary_metric_for("rung1")

    def test_raises_for_a_rung_not_in_the_table(self) -> None:
        with pytest.raises(ValueError, match="rung7"):
            primary_metric_for("rung7")
