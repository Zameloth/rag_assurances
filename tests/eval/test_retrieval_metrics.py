"""The eight (nine, by SPEC §12.6's own table) retrieval numbers — #35.

Every scenario builds `Candidate`/`RetrievalResult` objects directly rather than going
through Qdrant: the metrics operate on already-materialized retrieval output, so there is
no ranking or recall number here that needs the real engine (`CODING_STANDARDS.md`'s
in-memory-client rule is about *producing* candidates, not about scoring ones already in
hand).
"""

from __future__ import annotations

from pathlib import Path

from rag.eval.retrieval_metrics import (
    article_recall_working_set,
    floor_correctness_working_set,
    score_item,
    working_set,
)
from rag.eval.schema import GoldenItem, load_golden_set
from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.pipeline import ARTICLE_LEG, EXPANSION_POOL, FICHE_LEG, RetrievalResult
from rag.retrieval.short_circuit import ShortCircuitPath

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SET_PATH = REPO_ROOT / "eval" / "golden" / "golden-set.yaml"

SEARCH = frozenset({Provenance.SEARCH})
EXPANSION = frozenset({Provenance.EXPANSION})


def fiche_candidate(fiche_id: str, *, score: float = 1.0, text: str = "") -> Candidate:
    return Candidate(
        id=f"fiche:{fiche_id}",
        score=score,
        register=Register.FICHE,
        payload={"fiche_id": fiche_id, "text": text},
        provenance=SEARCH,
    )


def article_candidate(
    cid: str, *, score: float = 1.0, provenance: frozenset[Provenance] = SEARCH
) -> Candidate:
    return Candidate(
        id=f"article:{cid}",
        score=score,
        register=Register.ARTICLE,
        payload={"legiarti_cid": cid},
        provenance=provenance,
    )


def golden_item(
    id: str = "gs-001",
    *,
    history: tuple[dict[str, str], ...] = (),
    expected_state: str = "reponse",
    gold_fiches: tuple[str, ...] = (),
    gold_spans: tuple[str, ...] = (),
    gold_articles: tuple[str, ...] = (),
) -> GoldenItem:
    return GoldenItem(
        id=id,
        question="q",
        history=history,
        expected_state=expected_state,
        gold_fiches=gold_fiches,
        gold_spans=gold_spans,
        gold_articles=gold_articles,
        expected_points=(),
        tags=(),
    )


def result(
    contexts: list[Candidate],
    *,
    candidate_pools: dict[str, list[Candidate]] | None = None,
    path: ShortCircuitPath = ShortCircuitPath.NO_REFERENCE,
) -> RetrievalResult:
    return RetrievalResult(
        short_circuit_path=path, contexts=contexts, candidate_pools=candidate_pools or {}
    )


# --- working-set filters -----------------------------------------------------


class TestWorkingSet:
    def test_excludes_multi_turn(self) -> None:
        items = [golden_item("gs-001", gold_fiches=("F1",)), golden_item("gs-002", history=({"role": "user", "content": "hi"},), gold_fiches=("F2",))]
        assert [i.id for i in working_set(items)] == ["gs-001"]

    def test_excludes_items_with_no_gold_contexts_at_all(self) -> None:
        """`hors_corpus` items: empty gold_fiches and empty gold_articles both mean
        'nothing exists' (SPEC §12.4), not a retrieval target — nothing to score."""
        items = [golden_item("gs-001"), golden_item("gs-002", gold_fiches=("F1",))]
        assert [i.id for i in working_set(items)] == ["gs-002"]

    def test_keeps_items_with_only_gold_articles(self) -> None:
        items = [golden_item("gs-001", gold_articles=("CID1",))]
        assert [i.id for i in working_set(items)] == ["gs-001"]


class TestArticleRecallWorkingSet:
    def test_narrows_to_non_empty_gold_articles(self) -> None:
        items = [
            golden_item("gs-001", gold_fiches=("F1",)),  # reponse_sans_article-shaped
            golden_item("gs-002", gold_fiches=("F2",), gold_articles=("CID1",)),
        ]
        assert [i.id for i in article_recall_working_set(items)] == ["gs-002"]

    def test_still_excludes_multi_turn(self) -> None:
        items = [
            golden_item(
                "gs-001",
                history=({"role": "user", "content": "hi"},),
                gold_articles=("CID1",),
            )
        ]
        assert article_recall_working_set(items) == []


class TestFloorCorrectnessWorkingSet:
    def test_selects_only_reponse_sans_article(self) -> None:
        items = [
            golden_item("gs-001", expected_state="reponse", gold_fiches=("F1",)),
            golden_item(
                "gs-002", expected_state="reponse_sans_article", gold_fiches=("F2",)
            ),
        ]
        assert [i.id for i in floor_correctness_working_set(items)] == ["gs-002"]


# --- score_item: recall@4 / recall@10 ----------------------------------------


class TestFinalRecall:
    def test_fiche_recall_at_4_hit(self) -> None:
        item = golden_item(gold_fiches=("F1",))
        r = result([fiche_candidate("F1")])
        assert score_item(item, r).fiche_recall_at_4 == 1.0

    def test_fiche_recall_at_4_miss(self) -> None:
        item = golden_item(gold_fiches=("F1",))
        r = result([fiche_candidate("F2")])
        assert score_item(item, r).fiche_recall_at_4 == 0.0

    def test_fiche_recall_at_4_ignores_candidates_past_the_fourth_distinct_fiche(self) -> None:
        item = golden_item(gold_fiches=("F5",))
        contexts = [fiche_candidate(f"F{i}") for i in range(1, 5)] + [fiche_candidate("F5")]
        assert score_item(item, result(contexts)).fiche_recall_at_4 == 0.0

    def test_fiche_recall_at_10_reaches_past_the_final_cut(self) -> None:
        item = golden_item(gold_fiches=("F5",))
        contexts = [fiche_candidate(f"F{i}") for i in range(1, 5)] + [fiche_candidate("F5")]
        assert score_item(item, result(contexts)).fiche_recall_at_10 == 1.0

    def test_fiche_recall_dedupes_repeated_chunks_of_the_same_fiche(self) -> None:
        item = golden_item(gold_fiches=("F1", "F2"))
        contexts = [fiche_candidate("F1"), fiche_candidate("F1"), fiche_candidate("F1")]
        assert score_item(item, result(contexts)).fiche_recall_at_4 == 0.5

    def test_fiche_recall_is_none_when_gold_fiches_empty(self) -> None:
        item = golden_item(gold_articles=("CID1",))
        r = result([article_candidate("CID1")])
        assert score_item(item, r).fiche_recall_at_4 is None

    def test_article_recall_at_4_hit(self) -> None:
        item = golden_item(gold_articles=("CID1",))
        r = result([article_candidate("CID1")])
        assert score_item(item, r).article_recall_at_4 == 1.0

    def test_article_recall_partial_credit_across_multi_gold(self) -> None:
        item = golden_item(gold_articles=("CID1", "CID2"))
        r = result([article_candidate("CID1")])
        assert score_item(item, r).article_recall_at_4 == 0.5

    def test_article_recall_is_none_when_gold_articles_empty(self) -> None:
        item = golden_item(gold_fiches=("F1",))
        r = result([fiche_candidate("F1")])
        assert score_item(item, r).article_recall_at_4 is None
        assert score_item(item, r).article_recall_at_10 is None


# --- score_item: recall@candidate ("never found" vs "found then dropped") ---


class TestCandidateRecall:
    def test_found_in_pool_but_dropped_before_the_final_cut(self) -> None:
        item = golden_item(gold_articles=("CID1",))
        pools = {ARTICLE_LEG: [article_candidate("CID1")]}
        r = result([], candidate_pools=pools)
        scores = score_item(item, r)
        assert scores.article_recall_at_4 == 0.0
        assert scores.article_recall_at_candidate == 1.0

    def test_never_found_anywhere(self) -> None:
        item = golden_item(gold_articles=("CID1",))
        pools = {ARTICLE_LEG: [article_candidate("CID2")]}
        r = result([], candidate_pools=pools)
        assert score_item(item, r).article_recall_at_candidate == 0.0

    def test_article_candidate_pool_merges_expansion(self) -> None:
        item = golden_item(gold_articles=("CID1",))
        pools = {
            ARTICLE_LEG: [article_candidate("CID2")],
            EXPANSION_POOL: [article_candidate("CID1", provenance=frozenset({Provenance.EXPANSION}))],
        }
        r = result([], candidate_pools=pools)
        assert score_item(item, r).article_recall_at_candidate == 1.0

    def test_fiche_candidate_pool_is_the_fiche_leg_only(self) -> None:
        item = golden_item(gold_fiches=("F1",))
        pools = {FICHE_LEG: [fiche_candidate("F1")]}
        r = result([], candidate_pools=pools)
        assert score_item(item, r).fiche_recall_at_candidate == 1.0

    def test_none_on_the_short_circuit_path_with_no_candidate_pool(self) -> None:
        """The short-circuit skips search entirely (SPEC §9.1) — there is no candidate
        pool to diagnose 'found then dropped' against, which is a different fact from
        having searched and found nothing."""
        item = golden_item(gold_articles=("CID1",))
        r = result(
            [article_candidate("CID1")], candidate_pools={}, path=ShortCircuitPath.RESOLVED
        )
        scores = score_item(item, r)
        assert scores.article_recall_at_4 == 1.0
        assert scores.article_recall_at_candidate is None
        assert scores.fiche_recall_at_candidate is None


# --- score_item: zero-article rate and floor correctness ---------------------


class TestZeroArticlesAndFloor:
    def test_zero_articles_true_when_no_article_reached_the_final_contexts(self) -> None:
        item = golden_item(gold_fiches=("F1",), gold_articles=("CID1",))
        r = result([fiche_candidate("F1")])
        assert score_item(item, r).zero_articles is True

    def test_zero_articles_false_when_an_article_is_present(self) -> None:
        item = golden_item(gold_articles=("CID1",))
        r = result([article_candidate("CID1")])
        assert score_item(item, r).zero_articles is False

    def test_zero_articles_is_none_when_the_item_has_no_gold_articles(self) -> None:
        """Not applicable off the article-recall working set (SPEC §12.6: zero-article
        rate is reported over the 36/33-item subset expecting an article)."""
        item = golden_item(gold_fiches=("F1",))
        r = result([fiche_candidate("F1")])
        assert score_item(item, r).zero_articles is None

    def test_floor_correct_true_for_reponse_sans_article_with_no_article(self) -> None:
        item = golden_item(expected_state="reponse_sans_article", gold_fiches=("F1",))
        r = result([fiche_candidate("F1")])
        assert score_item(item, r).floor_correct is True

    def test_floor_correct_false_when_the_floor_was_not_held(self) -> None:
        item = golden_item(expected_state="reponse_sans_article", gold_fiches=("F1",))
        r = result([fiche_candidate("F1"), article_candidate("CID9")])
        assert score_item(item, r).floor_correct is False

    def test_floor_correct_is_none_off_the_reponse_sans_article_subset(self) -> None:
        item = golden_item(expected_state="reponse", gold_articles=("CID1",))
        r = result([article_candidate("CID1")])
        assert score_item(item, r).floor_correct is None


# --- score_item: span containment@4 ------------------------------------------


class TestSpanContainment:
    def test_full_credit_when_every_span_is_a_substring_of_a_top_fiche_chunk(self) -> None:
        item = golden_item(gold_spans=("l'assurance est obligatoire",))
        r = result([fiche_candidate("F1", text="Or, l'assurance est obligatoire pour tous.")])
        assert score_item(item, r).span_containment_at_4 == 1.0

    def test_zero_when_the_right_fiche_reached_top_4_but_not_the_span(self) -> None:
        """The 'right doc, wrong section' diagnostic (SPEC §12.6): the fiche is present,
        but this particular chunk doesn't carry the annotated span."""
        item = golden_item(gold_fiches=("F1",), gold_spans=("l'assurance est obligatoire",))
        r = result([fiche_candidate("F1", text="un chapitre sans rapport")])
        scores = score_item(item, r)
        assert scores.fiche_recall_at_4 == 1.0
        assert scores.span_containment_at_4 == 0.0

    def test_partial_credit_across_multiple_gold_spans(self) -> None:
        item = golden_item(gold_spans=("alpha", "beta"))
        r = result([fiche_candidate("F1", text="alpha only")])
        assert score_item(item, r).span_containment_at_4 == 0.5

    def test_none_when_no_gold_spans(self) -> None:
        item = golden_item(gold_fiches=("F1",))
        r = result([fiche_candidate("F1", text="anything")])
        assert score_item(item, r).span_containment_at_4 is None

    def test_ignores_article_register_text(self) -> None:
        item = golden_item(gold_spans=("l'assurance est obligatoire",))
        r = result([article_candidate("CID1")])
        assert score_item(item, r).span_containment_at_4 == 0.0


def test_short_circuit_path_is_recorded_on_every_item() -> None:
    item = golden_item(gold_articles=("CID1",))
    r = result([article_candidate("CID1")], path=ShortCircuitPath.RESOLVED)
    assert score_item(item, r).short_circuit_path == "resolved"


class TestWorkingSetAgainstTheRealGoldenSet:
    """SPEC §12.1/CONTEXT.md estimate the working set at "~44" single-turn items with gold
    contexts and "36" with non-empty `gold_articles`, fixed before the golden set was fully
    annotated (#45 completed it to 60 items). The finished set actually carries 40 and 33 —
    still comfortably inside the ADR-0011 sensitivity argument (net discordant pairs ≥ 4 at
    N=33-40 is the same order of magnitude as N=36), but this test pins the real number
    against silent drift and documents the gap rather than hard-coding the pre-registered
    estimate as if it still matched the finished set."""

    def test_actual_working_set_sizes(self) -> None:
        items = load_golden_set(GOLDEN_SET_PATH)
        assert len(working_set(items)) == 40
        assert len(article_recall_working_set(items)) == 33
        assert len(floor_correctness_working_set(items)) == 7
