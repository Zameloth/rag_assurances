"""Floor sweep selection logic (ADR-0027, #96), on synthetic scored pools — no reranker, no
Qdrant. Expected values are worked by hand from ADR-0027's rule, not recomputed through the
code under test."""

import json
from pathlib import Path

from rag.eval.floor_sweep import (
    FLOOR_GRID,
    ItemPool,
    PoolArticle,
    item_is_correct,
    pool_from_reranked,
    recompute_chosen_floor,
    run_sweep,
    select_floor,
    write_sweep_run,
)
from rag.eval.schema import GoldenItem
from rag.retrieval.candidates import Candidate, Provenance, Register


def _article(cid: str, score: float, *, expansion: bool = False, chunk: int = 0) -> PoolArticle:
    return PoolArticle(id=f"{cid}#{chunk}", legiarti_cid=cid, score=score, expansion=expansion)


def _article_item(item_id: str, gold: str, articles: list[PoolArticle]) -> ItemPool:
    return ItemPool(item_id=item_id, expected_state="reponse", gold_articles=(gold,), articles=tuple(articles))


def _sans_article_item(item_id: str, articles: list[PoolArticle]) -> ItemPool:
    return ItemPool(item_id=item_id, expected_state="reponse_sans_article", gold_articles=(), articles=tuple(articles))


def test_grid_is_0_to_0_5_in_steps_of_0_025() -> None:
    assert len(FLOOR_GRID) == 21
    assert FLOOR_GRID[0] == 0.0
    assert FLOOR_GRID[1] == 0.025
    assert FLOOR_GRID[-1] == 0.5


def test_article_item_is_correct_while_the_gold_article_clears_the_floor() -> None:
    pool = _article_item("a", "G", [_article("G", 0.3), _article("X", 0.9)])

    assert item_is_correct(pool, floor=0.3) is True
    assert item_is_correct(pool, floor=0.325) is False


def test_article_item_needs_the_gold_article_inside_the_four_article_slots() -> None:
    crowd = [_article(f"X{i}", 0.9 - i * 0.01) for i in range(4)]
    pool = _article_item("a", "G", [*crowd, _article("G", 0.2)])

    assert item_is_correct(pool, floor=0.0) is False


def test_sans_article_item_is_correct_only_when_the_floor_rejects_everything() -> None:
    pool = _sans_article_item("s", [_article("X", 0.4), _article("Y", 0.1)])

    assert item_is_correct(pool, floor=0.425) is True
    assert item_is_correct(pool, floor=0.4) is False


def test_expansion_preference_margin_stays_at_the_rung_5_value() -> None:
    # 5 articles clear the floor; the expansion-sourced G (0.55) gets +0.1 -> 0.65 and
    # outranks the 0.6 search hit, so it keeps a slot that plain score order would not give it.
    others = [_article("A", 0.9), _article("B", 0.8), _article("C", 0.7), _article("D", 0.6)]
    pool = _article_item("a", "G", [*others, _article("G", 0.55, expansion=True)])

    assert item_is_correct(pool, floor=0.5) is True


def test_select_floor_takes_the_highest_score() -> None:
    scores = {0.0: 1, 0.1: 3, 0.2: 2}

    assert select_floor(scores) == 0.1


def test_select_floor_breaks_ties_toward_the_higher_floor() -> None:
    scores = {0.1: 3, 0.2: 3, 0.3: 2}

    assert select_floor(scores) == 0.2


def test_select_floor_with_all_ties_picks_the_highest_grid_value() -> None:
    scores = {floor: 5 for floor in FLOOR_GRID}

    assert select_floor(scores) == 0.5


def test_sweep_picks_the_floor_separating_gold_from_junk() -> None:
    pools = [
        _article_item("a1", "G1", [_article("G1", 0.45), _article("X", 0.2)]),
        _article_item("a2", "G2", [_article("G2", 0.3)]),
        _sans_article_item("s1", [_article("Y", 0.2)]),
        _sans_article_item("s2", [_article("Z", 0.25)]),
    ]

    result = run_sweep(pools)

    # floors in (0.25, 0.3] keep both gold articles and reject both junk pools: 4/4.
    assert result.chosen_floor == 0.3
    assert result.no_workable_floor is False
    assert result.scores[0.3] == 4
    assert result.scores[0.325] == 3  # G2 (0.3) now rejected
    assert result.scores[0.0] == 2  # both sans-article items padded


def test_best_floor_of_zero_means_no_workable_floor() -> None:
    # Only gold-article items, every gold sits at score 0.0: nothing but floor 0.0 keeps them.
    pools = [_article_item("a1", "G1", [_article("G1", 0.0)])]

    result = run_sweep(pools)

    assert result.chosen_floor == 0.0
    assert result.no_workable_floor is True


def test_sweep_with_all_ties_chooses_the_highest_floor_without_flagging() -> None:
    pools = [_sans_article_item("s1", [])]  # empty pool: rejected at every floor

    result = run_sweep(pools)

    assert result.chosen_floor == 0.5
    assert result.no_workable_floor is False


def test_short_circuited_items_are_excluded_from_every_score() -> None:
    pools = [
        _article_item("a1", "G1", [_article("G1", 0.4)]),
        ItemPool(item_id="lk", expected_state="reponse", gold_articles=("G9",), articles=(), short_circuit=True),
    ]

    result = run_sweep(pools)

    assert result.scores[0.4] == 1
    assert result.skipped_items == ("lk",)


def test_output_is_enough_to_recompute_the_pick_without_the_reranker() -> None:
    pools = [
        _article_item("a1", "G1", [_article("G1", 0.45, expansion=True), _article("X", 0.2)]),
        _sans_article_item("s1", [_article("Y", 0.2)]),
    ]
    result = run_sweep(pools)

    payload = result.to_json()

    assert payload["chosen_floor"] == result.chosen_floor
    assert recompute_chosen_floor(payload) == result.chosen_floor
    assert set(payload["items"][0]["pool"][0]) == {"id", "legiarti_cid", "score", "expansion"}
    assert payload["per_floor"][0]["floor"] == 0.0


def test_recompute_follows_the_cached_pools_not_the_stored_pick() -> None:
    pools = [_article_item("a1", "G1", [_article("G1", 0.45)])]
    payload = run_sweep(pools).to_json()
    payload["chosen_floor"] = 0.0  # tampered

    assert recompute_chosen_floor(payload) == 0.45


# --- pool capture and output file ------------------------------------------------------


def _candidate(point_id: str, cid: str, score: float, register: Register, *, expansion: bool = False) -> Candidate:
    key = "legiarti_cid" if register is Register.ARTICLE else "fiche_id"
    provenance = Provenance.EXPANSION if expansion else Provenance.SEARCH
    return Candidate(
        id=point_id, score=score, register=register, payload={key: cid, "text": "texte"}, provenance=frozenset({provenance})
    )


def test_pool_from_reranked_keeps_articles_only_and_drops_text() -> None:
    item = GoldenItem(
        id="gs-001", question="q", history=(), expected_state="reponse", gold_fiches=("F1",),
        gold_spans=(), gold_articles=("G",), expected_points=("p",), tags=(),
    )
    reranked = [
        _candidate("f#0", "F1", 0.9, Register.FICHE),
        _candidate("a#0", "G", 0.6, Register.ARTICLE, expansion=True),
    ]

    pool = pool_from_reranked(item, reranked, short_circuit=False)

    assert pool.item_id == "gs-001"
    assert pool.gold_articles == ("G",)
    assert pool.articles == (PoolArticle(id="a#0", legiarti_cid="G", score=0.6, expansion=True),)


def test_write_sweep_run_writes_a_json_file_named_by_timestamp(tmp_path: Path) -> None:
    result = run_sweep([_sans_article_item("s1", [_article("Y", 0.2)])])

    path = write_sweep_run(result, tmp_path / "runs", stamp="20261012T100000Z")

    assert path.name == "floor-sweep-20261012T100000Z.json"
    assert recompute_chosen_floor(json.loads(path.read_text(encoding="utf-8"))) == result.chosen_floor
