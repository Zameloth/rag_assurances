"""Floor sweep: pure selection logic for ADR-0027 (#84, #96).

The reranker runs once per calibration item (`scripts/run_floor_sweep.py`); what comes out of
it is each item's scored **article pool** — ids, scores and provenance, no text. Everything
here is offline and deterministic over those pools: `assemble_quota` is evaluated at every
grid value and ADR-0027's selection rule picks the floor. `SweepResult.to_json()` carries the
pools themselves, so `recompute_chosen_floor` can redo the pick from the written file alone.

**Selection rule (ADR-0027 §3)**: a floor's score is the number of correct article-side
decisions — article items with a gold article in the quota's article slots, plus
`reponse_sans_article` items where the floor rejects every article. Highest score wins; ties
go to the **higher** floor (SPEC §9.5, "reject rather than pad"). A chosen floor of 0.0 means
"no workable floor". Items that took the short-circuit path have no pool and are skipped.

Only the article side is pooled: fiche slots are filled independently of articles
(`assemble_quota`), so they cannot change an article-side decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.eval.schema import GoldenItem
from rag.retrieval.candidates import Candidate, Provenance, Register
from rag.retrieval.quota import EXPANSION_PREFERENCE_MARGIN, NO_ARTICLE_MARKER_ID, assemble_quota

__all__ = [
    "FLOOR_GRID",
    "ItemPool",
    "PoolArticle",
    "SweepResult",
    "item_is_correct",
    "pool_from_reranked",
    "recompute_chosen_floor",
    "run_sweep",
    "select_floor",
    "write_sweep_run",
]

# ADR-0027 §2: 0.00-0.50 inclusive, step 0.025. Rounded so every value is the literal a
# human would write (0.3, not 0.30000000000000004) and `score >= floor` behaves as read.
FLOOR_GRID: tuple[float, ...] = tuple(round(step * 0.025, 3) for step in range(21))


@dataclass(frozen=True)
class PoolArticle:
    """One reranked article chunk: no text, just what `assemble_quota` reads."""

    id: str
    legiarti_cid: str
    score: float
    expansion: bool  # `Provenance.EXPANSION in candidate.provenance`

    def to_candidate(self) -> Candidate:
        provenance = Provenance.EXPANSION if self.expansion else Provenance.SEARCH
        return Candidate(
            id=self.id,
            score=self.score,
            register=Register.ARTICLE,
            payload={"legiarti_cid": self.legiarti_cid},
            provenance=frozenset({provenance}),
        )


@dataclass(frozen=True)
class ItemPool:
    item_id: str
    expected_state: str
    gold_articles: tuple[str, ...]
    articles: tuple[PoolArticle, ...]
    short_circuit: bool = False


def item_is_correct(
    pool: ItemPool, *, floor: float, expansion_preference_margin: float = EXPANSION_PREFERENCE_MARGIN
) -> bool:
    """ADR-0027 §3's per-item decision at one floor."""
    result = assemble_quota(
        [article.to_candidate() for article in pool.articles],
        relevance_floor=floor,
        expansion_preference_margin=expansion_preference_margin,
    )
    slot_cids = {
        str(candidate.payload["legiarti_cid"])
        for candidate in result.contexts
        if candidate.register is Register.ARTICLE and candidate.id != NO_ARTICLE_MARKER_ID
    }
    if pool.expected_state == "reponse_sans_article":
        return not slot_cids
    return bool(slot_cids & set(pool.gold_articles))


def select_floor(scores: Mapping[float, int]) -> float:
    """Highest score; ties resolved to the higher floor."""
    return max(scores, key=lambda floor: (scores[floor], floor))


@dataclass(frozen=True)
class SweepResult:
    pools: tuple[ItemPool, ...]
    expansion_preference_margin: float
    scores: dict[float, int]
    per_item: dict[float, dict[str, bool]]
    chosen_floor: float
    skipped_items: tuple[str, ...]

    @property
    def no_workable_floor(self) -> bool:
        return self.chosen_floor == 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "rule": "ADR-0027: maximise correct article-side decisions; ties -> higher floor; 0.0 = no workable floor",
            "grid": list(FLOOR_GRID),
            "expansion_preference_margin": self.expansion_preference_margin,
            "chosen_floor": self.chosen_floor,
            "no_workable_floor": self.no_workable_floor,
            "skipped_items": list(self.skipped_items),
            "per_floor": [
                {"floor": floor, "correct": self.scores[floor], "per_item": self.per_item[floor]}
                for floor in FLOOR_GRID
            ],
            "items": [
                {
                    "item_id": pool.item_id,
                    "expected_state": pool.expected_state,
                    "gold_articles": list(pool.gold_articles),
                    "short_circuit": pool.short_circuit,
                    "pool": [
                        {"id": a.id, "legiarti_cid": a.legiarti_cid, "score": a.score, "expansion": a.expansion}
                        for a in pool.articles
                    ],
                }
                for pool in self.pools
            ],
        }


def run_sweep(
    pools: Sequence[ItemPool],
    *,
    grid: Sequence[float] = FLOOR_GRID,
    expansion_preference_margin: float = EXPANSION_PREFERENCE_MARGIN,
) -> SweepResult:
    scored = [pool for pool in pools if not pool.short_circuit]
    per_item = {
        floor: {
            pool.item_id: item_is_correct(pool, floor=floor, expansion_preference_margin=expansion_preference_margin)
            for pool in scored
        }
        for floor in grid
    }
    scores = {floor: sum(decisions.values()) for floor, decisions in per_item.items()}
    return SweepResult(
        pools=tuple(pools),
        expansion_preference_margin=expansion_preference_margin,
        scores=scores,
        per_item=per_item,
        chosen_floor=select_floor(scores),
        skipped_items=tuple(pool.item_id for pool in pools if pool.short_circuit),
    )


def pool_from_reranked(item: GoldenItem, reranked: Sequence[Candidate], *, short_circuit: bool) -> ItemPool:
    """Cache one item's scored article pool from the reranker's output: ids, scores and
    provenance only — never the text."""
    return ItemPool(
        item_id=item.id,
        expected_state=item.expected_state,
        gold_articles=item.gold_articles,
        articles=tuple(
            PoolArticle(
                id=candidate.id,
                legiarti_cid=str(candidate.payload["legiarti_cid"]),
                score=candidate.score,
                expansion=Provenance.EXPANSION in candidate.provenance,
            )
            for candidate in reranked
            if candidate.register is Register.ARTICLE
        ),
        short_circuit=short_circuit,
    )


def write_sweep_run(result: SweepResult, runs_dir: Path, *, stamp: str) -> Path:
    """`eval/runs/floor-sweep-<ts>.json` (SPEC §12.11's per-item-scores-to-git posture)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"floor-sweep-{stamp}.json"
    path.write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def recompute_chosen_floor(payload: Mapping[str, Any]) -> float:
    """Redo the pick from a written sweep file — cached pools only, no reranker."""
    pools = [
        ItemPool(
            item_id=item["item_id"],
            expected_state=item["expected_state"],
            gold_articles=tuple(item["gold_articles"]),
            articles=tuple(PoolArticle(**article) for article in item["pool"]),
            short_circuit=item["short_circuit"],
        )
        for item in payload["items"]
    ]
    return run_sweep(
        pools, grid=payload["grid"], expansion_preference_margin=payload["expansion_preference_margin"]
    ).chosen_floor
