#!/usr/bin/env python3
"""Run ladder rungs 1-3 end to end and record their verdicts (SPEC §12.7, ADR-0015/16/17, #40).

Rung 1 is the naive baseline (dense-only, no expansion, no rerank, top-8) — SPEC §12.7 calls it
"a reference floor, not a comparison", so it is run and persisted like every other rung but never
handed to `compare.py`. Rungs 2 and 3 each change exactly one variable from the rung below them
(the hybrid sparse leg, then `<dc:source>` expansion — ADR-0016/ADR-0017), so each is compared
against that rung, not against rung 1: `compare_runs(rung1, rung2, ...)` for rung 2's verdict,
`compare_runs(rung2, rung3, ...)` for rung 3's. All three runs execute the *same* golden-set items
against the *same* incumbent collections (`articles`/`fiches` — no alias flip, unlike
`run_ab_pilot.py`'s challenger collections), so the pairing `compare_runs` requires is automatic.

**The primary metric is never a command-line argument here either** — same discipline as
`compare_runs.py` (#36/#37): each verdict's primary is resolved from `rag.eval.ladder_registry`
via `primary_metric_for`, not typed in after seeing rung 2 or rung 3's numbers.

The ladder is fully deterministic and costs no API spend (SPEC §12.5): no LLM sits in the loop for
rungs 1-3, so the only per-run cost is embedding the golden set's single-turn queries through the
resident BGE-M3 model.

    uv run python scripts/run_ladder.py
    uv run python scripts/run_ladder.py --no-sync
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from rag.config import load_settings
from rag.eval.compare import ComparisonReport, compare_runs, write_comparison
from rag.eval.ladder_registry import primary_metric_for
from rag.eval.langfuse_sync import sync_retrieval_dataset
from rag.eval.retrieval_run import RetrievalRun
from rag.eval.run_experiment import run_retrieval_ladder
from rag.ingest.embedder import MODEL_ID as EMBEDDER_MODEL_ID
from rag.retrieval.expansion import EXPANSION_CAP, EXPANSION_FICHE_DEPTH
from rag.retrieval.fusion import ARTICLE_LEG_WEIGHTS, FICHE_LEG_WEIGHTS
from rag.retrieval.lookup import load_lookup_keys
from rag.retrieval.pipeline import LEG_CANDIDATE_LIMIT, TOP_K

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SET_PATH = REPO_ROOT / "eval" / "golden" / "golden-set.yaml"
RUNS_DIR = REPO_ROOT / "eval" / "runs"

# SPEC §12.7's first three rows — rung 4 onward is a later ticket's job.
RUNGS = ("rung1", "rung2", "rung3")

# `(rung, rung-below)` pairs SPEC §12.7's ladder actually compares — rung 1 has no pair,
# since it is the floor everything else is read against, not a comparison itself.
COMPARISON_PAIRS = (("rung2", "rung1"), ("rung3", "rung2"))


def _retrieval_config(rung: str) -> dict[str, object]:
    """The caller's own pinned description of what `rung` actually runs (SPEC §12.11) — read
    off `rag.retrieval.pipeline`'s and friends' named constants, since only this script knows
    which rung is being asked to run."""
    config: dict[str, object] = {
        "embedder": EMBEDDER_MODEL_ID,
        "leg_candidate_limit": LEG_CANDIDATE_LIMIT,
        "top_k": TOP_K,
    }
    if rung == "rung1":
        return config
    config["fiche_leg_weights"] = {"dense": FICHE_LEG_WEIGHTS.dense, "sparse": FICHE_LEG_WEIGHTS.sparse}
    config["article_leg_weights"] = {"dense": ARTICLE_LEG_WEIGHTS.dense, "sparse": ARTICLE_LEG_WEIGHTS.sparse}
    if rung == "rung2":
        return config
    config["expansion_depth"] = EXPANSION_FICHE_DEPTH
    config["expansion_cap"] = EXPANSION_CAP
    return config


def _print_metrics(report: ComparisonReport) -> None:
    print(f"primary: {report.primary_metric}")
    for metric, comparison in report.metrics.items():
        print(
            f"  {metric:<28} n={comparison.n_paired:>2}  improved={comparison.improved:>2}  "
            f"regressed={comparison.regressed:>2}  net={comparison.net_discordant:+d}  "
            f"sign-test p={comparison.sign_test_p:.3f}"
        )
        if comparison.regressed_items:
            print(f"    regressed: {', '.join(comparison.regressed_items)}")
    verdict = report.verdict
    outcome = "ADOPT" if verdict.adopt else "KEEP INCUMBENT (inconclusive resolves to no change)"
    print(f"VERDICT: {outcome} (net on primary = {verdict.net_discordant_primary:+d})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--no-sync", action="store_true", help="skip syncing the golden set to Langfuse first"
    )
    args = parser.parse_args(argv)

    settings = load_settings()  # also loads .env into the process environment for get_client()
    if not args.no_sync:
        print(f"syncing {GOLDEN_SET_PATH} -> Langfuse retrieval dataset ...")
        sync_retrieval_dataset(GOLDEN_SET_PATH)

    client = QdrantClient(settings.qdrant_url)
    from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    runs: dict[str, RetrievalRun] = {}
    try:
        lookup_keys = load_lookup_keys(client)
        for rung in RUNGS:
            run_id = f"{rung}-{stamp}"
            print(f"=== {rung} ===")
            runs[rung] = run_retrieval_ladder(
                client=client,
                embed=embed_batch,
                lookup_keys=lookup_keys,
                arm=rung,
                rung=rung,
                run_id=run_id,
                repo_root=REPO_ROOT,
                golden_set_path=GOLDEN_SET_PATH,
                runs_dir=RUNS_DIR,
                retrieval_config=_retrieval_config(rung),
            )
            print(f"  written to eval/runs/{run_id}.json ({len(runs[rung].items)} item(s))")

        for rung, below in COMPARISON_PAIRS:
            print(f"\n=== {rung} vs {below} ===")
            report = compare_runs(runs[below], runs[rung], primary_metric_for(rung))
            verdict_path = RUNS_DIR / f"{rung}-verdict-{stamp}.json"
            write_comparison(report, verdict_path)
            _print_metrics(report)
            print(f"verdict written to {verdict_path}")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
