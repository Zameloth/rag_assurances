#!/usr/bin/env python3
"""Sweep `ARTICLE_RELEVANCE_FLOOR` over the floor-calibration set (ADR-0027, #84, #96).

**A dev tool, run by hand — and only after ADR-0027 is merged.** Needs a live Qdrant with the
incumbent collections and the local reranker (CPU-heavy), like the ladder scripts. The
calibration file is read **only here**: no ladder script touches it (guarded by
`tests/eval/test_floor_sweep_isolation.py`).

For each calibration item the rung-5 pipeline runs once (`retrieve_rung5`, every setting at
its rung-5 value) with a `rerank_fn` that records the reranked pool; the floor is then swept
offline over those cached pools (`rag.eval.floor_sweep`) and the result — pools, per-floor
scores, chosen floor — is written to `eval/runs/floor-sweep-<ts>.json`, enough to recompute
the pick without the reranker.

    uv run python scripts/run_floor_sweep.py [--calibration-set PATH] [--golden-set PATH]

A chosen floor of 0.0 means "no workable floor" (ADR-0027 §3): the script says so and the
ticket stops there.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from rag.config import load_settings
from rag.eval.floor_sweep import ItemPool, pool_from_reranked, run_sweep, write_sweep_run
from rag.eval.schema import GoldenItem, load_golden_set
from rag.eval.validate import validate_floor_calibration_set_against_corpus
from rag.ingest.upsert import EmbedFn
from rag.retrieval.candidates import Candidate
from rag.retrieval.lookup import load_lookup_keys
from rag.retrieval.pipeline import retrieve_rung5
from rag.retrieval.rerank import rerank
from rag.retrieval.short_circuit import ShortCircuitPath

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_SET = REPO_ROOT / "eval" / "golden" / "floor-calibration.yaml"
DEFAULT_GOLDEN_SET = REPO_ROOT / "eval" / "golden" / "golden-set.yaml"
RUNS_DIR = REPO_ROOT / "eval" / "runs"
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"


def capture_item_pool(client: QdrantClient, embed: EmbedFn, lookup_keys: set[str], item: GoldenItem) -> ItemPool:
    """Run rung 5 for one item, recording the reranker's output. Rung 5's own quota result
    is ignored (it ran at the default floor); only the reranked pool matters."""
    captured: list[Candidate] = []

    def recording_rerank(query: str, candidates: list[Candidate]) -> list[Candidate]:
        reranked = rerank(query, candidates)
        captured.extend(reranked)
        return reranked

    result = retrieve_rung5(client, embed, item.question, lookup_keys, rerank_fn=recording_rerank)
    return pool_from_reranked(
        item, captured, short_circuit=result.short_circuit_path is ShortCircuitPath.RESOLVED
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration-set", type=Path, default=DEFAULT_CALIBRATION_SET)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    args = parser.parse_args(argv)

    items = load_golden_set(args.calibration_set)
    if not items:
        print(f"{args.calibration_set}: no items — nothing to sweep", file=sys.stderr)
        return 1
    validate_floor_calibration_set_against_corpus(
        items, golden_items=load_golden_set(args.golden_set), fiches_dir=FICHES_DIR, articles_path=ARTICLES_PATH
    )

    settings = load_settings()
    from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

    client = QdrantClient(settings.qdrant_url)
    try:
        lookup_keys = set(load_lookup_keys(client))
        pools = []
        for index, item in enumerate(items, start=1):
            print(f"[{index}/{len(items)}] {item.id}")
            pools.append(capture_item_pool(client, embed_batch, lookup_keys, item))
    finally:
        client.close()

    result = run_sweep(pools)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = write_sweep_run(result, RUNS_DIR, stamp=stamp)
    for floor, score in result.scores.items():
        print(f"  floor {floor:.3f}  correct {score}")
    print(f"chosen floor: {result.chosen_floor}  ->  {path}")
    if result.no_workable_floor:
        print("best floor is 0.0: NO WORKABLE FLOOR (ADR-0027 §3) — record it and stop the ticket")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
