#!/usr/bin/env python3
"""Run ladder rungs 4-6 end to end and record their verdicts (SPEC §12.7, ADR-0018/19/22, #41).

Continues `scripts/run_ladder.py`'s own discipline exactly: each rung changes exactly one
variable from the rung immediately below it (ADR-0023's own framing — "#40's own acceptance
criterion"), so rung 4 is compared against rung 3, rung 5 against rung 4, rung 6 against
rung 5, never against rung 1. Rung 3's own run is **not** re-executed here — #40 already
committed it (`eval/runs/rung3-20260916T181814Z.json`), `compare_runs` pairs by item id, and
nothing about rung 3's code path has changed since (ADR-0023). Loading it, not re-running
it, is the same "a decision input has to survive past the process that produced it" posture
`rag.eval.resident_memory`'s own module docstring already states for the rung 6 RAM report.

**Rungs 4 and 5 run against the incumbent `articles`/`fiches` collections**, same as rungs
1-3 — `retrieve_rung4`/`retrieve_rung5` change only the pipeline, not the index.

**Rung 6 is index-bearing** (ADR-0022): its dense half comes from a different embedder, on
a different pair of physical collections (`articles__e5-m3__c512__v1`/
`fiches__e5-m3__c512__v1`, already built by #39). Reaching them means temporarily flipping
`ARTICLES_ALIAS`/`FICHES_ALIAS` — every retrieval call in this codebase reads those two
stable names, never a physical collection, so there is no other seam. `scripts/
run_ab_pilot.py`'s own pattern is followed exactly: the alias is restored to the incumbent
arm in a `finally`, success or failure, so adopting rung 6 stays the deliberate, separate
step SPEC §12.8 already requires of every challenger arm — never a side effect of having
measured it. Rung 6 runs through `pipeline_arm="rung5"` (SPEC's row 6 holds the reranker and
quota fixed at rung 5's config and swaps only the embedder — "embedder A/B") with `rag.
ingest.e5_embedder.query_embed_batch` as `embed`, exactly the seam ADR-0022 named for this
ticket to consume.

Rung 4's resident-memory figure (`scripts/measure_reranker_ram.py`,
`eval/runs/rung4-resident-memory.json`) and rung 6's (`scripts/measure_e5_ram.py`,
`eval/runs/rung6-resident-memory.json`) are both read and printed alongside their quality
verdicts here, never re-measured — SPEC §14.4/§17.2's and §15.7's RAM questions are answered
by those dedicated, query-shaped, build-free scripts, not by this one.

**Two things discovered running this for real, both worked around here rather than in the
shared modules they live in:**

1. `rag.ingest.embedder`/`rag.ingest.e5_embedder`/`rag.retrieval.rerank`'s lazy
   `@lru_cache`/`@cache`-backed model loaders are not safe against concurrent first calls —
   several of Langfuse's concurrent `task()` invocations racing to construct the same
   cached model at once produces a corrupted or (on a CUDA build) half-initialized model.
   Every model this script needs is warmed by one serial call before any concurrent rung
   runs, so no `task()` invocation is ever the one paying for construction.
2. `run_retrieval_ladder`'s default concurrency (`rag.eval.run_experiment.MAX_CONCURRENCY`,
   tuned for OpenRouter rate limits — SPEC §12.5/ADR-0012) is too high for a `task` that
   does local CPU-bound cross-encoder inference: several reranker calls in flight at once
   compete for the same process's CPU and RAM (SPEC §14.4's own "hundreds of MB" of
   transient activation per call). Rungs 4-6 pass a much lower `RERANKER_MAX_CONCURRENCY`
   instead.

    uv run python scripts/run_ladder_rungs_4_6.py
    uv run python scripts/run_ladder_rungs_4_6.py --no-sync
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from rag.config import load_settings
from rag.eval.compare import ComparisonReport, compare_runs, write_comparison
from rag.eval.ladder_registry import primary_metric_for
from rag.eval.langfuse_sync import sync_retrieval_dataset
from rag.eval.retrieval_run import RetrievalRun, load_run
from rag.eval.run_experiment import run_retrieval_ladder
from rag.ingest.arms import ARTICLES_ALIAS, FICHES_ALIAS, flip_alias
from rag.ingest.e5_arm import ARTICLES_E5_ARM, FICHES_E5_ARM
from rag.ingest.e5_embedder import MODEL_ID as E5_MODEL_ID
from rag.ingest.embedder import MODEL_ID as EMBEDDER_MODEL_ID
from rag.ingest.pipeline import ARTICLES_ARM, FICHES_ARM
from rag.retrieval.expansion import EXPANSION_CAP, EXPANSION_FICHE_DEPTH
from rag.retrieval.fusion import ARTICLE_LEG_WEIGHTS, FICHE_LEG_WEIGHTS
from rag.retrieval.lookup import load_lookup_keys
from rag.retrieval.pipeline import LEG_CANDIDATE_LIMIT, TOP_K
from rag.retrieval.quota import (
    ARTICLE_QUOTA,
    ARTICLE_RELEVANCE_FLOOR,
    EXPANSION_PREFERENCE_MARGIN,
    FICHE_QUOTA,
)
from rag.retrieval.rerank import DEFAULT_RERANKER_MODEL, RerankerBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SET_PATH = REPO_ROOT / "eval" / "golden" / "golden-set.yaml"
RUNS_DIR = REPO_ROOT / "eval" / "runs"

# #40's own committed rung-3 run — the "rung immediately below" rung 4 is compared against
# (ADR-0023). Not re-run here; see the module docstring.
RUNG3_RUN_PATH = RUNS_DIR / "rung3-20260916T181814Z.json"

RUNG4_RESIDENT_MEMORY_PATH = RUNS_DIR / "rung4-resident-memory.json"
RUNG6_RESIDENT_MEMORY_PATH = RUNS_DIR / "rung6-resident-memory.json"

# Local CPU-bound reranker inference, not an API call — see the module docstring's second
# "discovered running this for real" note. Low enough that several concurrent reranker
# calls' transient activation memory (SPEC §14.4) doesn't compete for the same handful of
# CPU cores this dev box has.
RERANKER_MAX_CONCURRENCY = 3

# A representative single question — the same warm-up shape `scripts/measure_e5_ram.py` and
# `scripts/measure_reranker_ram.py` already use (batch of one, not a bulk-ingest batch).
_WARMUP_QUERY = "Quel est le délai de résiliation d'un contrat d'assurance habitation ?"


def _shared_retrieval_config() -> dict[str, object]:
    """The fusion/expansion config every rung from 2 onward already pins (`scripts/
    run_ladder.py`'s own `_retrieval_config` for those rungs) — restated here since rung 4
    onward is a separate script."""
    return {
        "embedder": EMBEDDER_MODEL_ID,
        "leg_candidate_limit": LEG_CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "fiche_leg_weights": {"dense": FICHE_LEG_WEIGHTS.dense, "sparse": FICHE_LEG_WEIGHTS.sparse},
        "article_leg_weights": {"dense": ARTICLE_LEG_WEIGHTS.dense, "sparse": ARTICLE_LEG_WEIGHTS.sparse},
        "expansion_depth": EXPANSION_FICHE_DEPTH,
        "expansion_cap": EXPANSION_CAP,
    }


def _rung4_config() -> dict[str, object]:
    config = _shared_retrieval_config()
    config["reranker_model"] = DEFAULT_RERANKER_MODEL
    config["reranker_backend"] = RerankerBackend.FP32.value
    return config


def _rung5_config() -> dict[str, object]:
    config = _rung4_config()
    config["fiche_quota"] = FICHE_QUOTA
    config["article_quota"] = ARTICLE_QUOTA
    config["relevance_floor"] = ARTICLE_RELEVANCE_FLOOR
    config["expansion_preference_margin"] = EXPANSION_PREFERENCE_MARGIN
    return config


def _rung6_config() -> dict[str, object]:
    config = _rung5_config()
    config["dense_embedder"] = E5_MODEL_ID
    config["sparse_embedder"] = EMBEDDER_MODEL_ID
    config["collection_arm"] = {"articles": ARTICLES_E5_ARM, "fiches": FICHES_E5_ARM}
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


def _print_resident_memory(path: Path) -> None:
    if not path.exists():
        print(f"  (no resident-memory report at {path})")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"  resident memory: {report['rss_mb']:.0f} MB ({', '.join(report['models'])})")


def _warm_up_models() -> None:
    """One serial call per model this script's rungs load, before any concurrent rung run
    starts — see the module docstring's first "discovered running this for real" note.
    `rag.retrieval.rerank._load_reranker`/`rag.ingest.embedder._model`/`rag.ingest.
    e5_embedder._model` are all `@lru_cache`/`@cache`-backed lazy singletons, and none of
    them are safe against several threads racing to construct the same cache miss at once
    (Langfuse's concurrent `task()` calls do exactly that). Warming each one here, before
    `run_retrieval_ladder` ever runs concurrently, means no `task()` invocation is ever the
    one paying for construction."""
    from rag.ingest.e5_embedder import query_embed_batch
    from rag.ingest.embedder import embed_batch
    from rag.retrieval.candidates import Candidate, Provenance, Register
    from rag.retrieval.rerank import rerank

    print("warming up models (BGE-M3, e5, reranker) ...")
    embed_batch([_WARMUP_QUERY])
    query_embed_batch([_WARMUP_QUERY])
    warmup_candidate = Candidate(
        id="warmup",
        score=0.0,
        register=Register.ARTICLE,
        payload={"text": "Le contrat peut être résilié chaque année moyennant un préavis."},
        provenance=frozenset({Provenance.SEARCH}),
    )
    rerank(_WARMUP_QUERY, [warmup_candidate])
    print("  done")


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
    from rag.ingest.e5_embedder import query_embed_batch
    from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

    _warm_up_models()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    runs: dict[str, RetrievalRun] = {"rung3": load_run(RUNG3_RUN_PATH)}
    try:
        lookup_keys = load_lookup_keys(client)

        for rung, config_fn in (("rung4", _rung4_config), ("rung5", _rung5_config)):
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
                retrieval_config=config_fn(),
                max_concurrency=RERANKER_MAX_CONCURRENCY,
            )
            print(f"  written to eval/runs/{run_id}.json ({len(runs[rung].items)} item(s))")

        print("=== rung6 ===")
        try:
            flip_alias(client, ARTICLES_ALIAS, ARTICLES_E5_ARM)
            flip_alias(client, FICHES_ALIAS, FICHES_E5_ARM)
            print(f"[{ARTICLES_ALIAS}] -> {ARTICLES_E5_ARM}, [{FICHES_ALIAS}] -> {FICHES_E5_ARM}")
            e5_lookup_keys = load_lookup_keys(client)
            run_id = f"rung6-{stamp}"
            runs["rung6"] = run_retrieval_ladder(
                client=client,
                embed=query_embed_batch,
                lookup_keys=e5_lookup_keys,
                arm="rung6",
                rung="rung6",
                pipeline_arm="rung5",
                run_id=run_id,
                repo_root=REPO_ROOT,
                golden_set_path=GOLDEN_SET_PATH,
                runs_dir=RUNS_DIR,
                retrieval_config=_rung6_config(),
                max_concurrency=RERANKER_MAX_CONCURRENCY,
            )
            print(f"  written to eval/runs/{run_id}.json ({len(runs['rung6'].items)} item(s))")
        finally:
            flip_alias(client, ARTICLES_ALIAS, ARTICLES_ARM)
            flip_alias(client, FICHES_ALIAS, FICHES_ARM)
            print(f"[{ARTICLES_ALIAS}] restored -> {ARTICLES_ARM}, [{FICHES_ALIAS}] restored -> {FICHES_ARM}")

        for rung, below in (("rung4", "rung3"), ("rung5", "rung4"), ("rung6", "rung5")):
            print(f"\n=== {rung} vs {below} ===")
            report = compare_runs(runs[below], runs[rung], primary_metric_for(rung))
            verdict_path = RUNS_DIR / f"{rung}-verdict-{stamp}.json"
            write_comparison(report, verdict_path)
            _print_metrics(report)
            print(f"verdict written to {verdict_path}")

        print("\n=== resident memory (read, not re-measured) ===")
        print("rung4:")
        _print_resident_memory(RUNG4_RESIDENT_MEMORY_PATH)
        print("rung6:")
        _print_resident_memory(RUNG6_RESIDENT_MEMORY_PATH)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
