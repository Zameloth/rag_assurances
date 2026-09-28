#!/usr/bin/env python3
"""Run the generation eval end to end and record the verdict (SPEC §12.9, §12.10,
§12.11, §12.12, ADR-0009, ADR-0025, #46, #47).

Five metrics. State accuracy, citation validity and citation correctness cost nothing and
cannot drift (SPEC §12.9); faithfulness and point coverage are judged by `JUDGE_MODEL`
(#47), on pinned OpenRouter routing, with the resolved provider recorded in the run header.
`--no-judge` runs the deterministic three alone. Every item runs the full chain
(condensation when it carries history, then retrieval, then generation), plus two judge
calls unless `--no-judge`, so this is not free the way `run_ladder.py`'s rungs 1-3 are.

**Trust the judged numbers only for a judge configuration that passed calibration**
(`scripts/run_judge_calibration.py`, SPEC §12.10) — same judge model, same prompt
languages, same resolved provider.

**Runs on top of the ladder-winning arm** (ADR-0024: rung 1 stands — reranking and the quota
guard both failed their adoption bar, and the e5 embedder A/B was a wash), not whichever rung
happened to run most recently — `GENERATION_RETRIEVAL_ARM` is a named constant for exactly that
reason, shared with the calibration authoring helper.

**Citation correctness is a distinct number from article recall@4, read together, not summed**
(SPEC §12.9): recall says the gold article reached the prompt (a fact from `run_ladder.py`'s
own runs against the working set), correctness says the model actually cited it (this run,
against the full 60-item set). Recall high + correctness low is a generation failure; both low
is a retrieval failure. This script's own summary prints a reminder of that pairing rather than
computing recall itself — recall belongs to the ladder's own dataset and run, a different N.

    uv run python scripts/run_generation_eval.py
    uv run python scripts/run_generation_eval.py --no-sync
    uv run python scripts/run_generation_eval.py --no-judge
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from qdrant_client import QdrantClient

from rag.condensation.chain import make_condense_fn
from rag.config import load_settings
from rag.eval.generation_run import GenerationRun
from rag.eval.judge_chain import make_judge
from rag.eval.langfuse_sync import sync_generation_dataset
from rag.eval.paths import GOLDEN_SET_PATH, REPO_ROOT, RUNS_DIR
from rag.eval.run_generation_experiment import (
    MAX_CONCURRENCY,
    JudgeRunError,
    run_generation_eval,
)
from rag.generation.chain import make_generate_fn
from rag.ingest.embedder import MODEL_ID as EMBEDDER_MODEL_ID
from rag.pipeline import GENERATION_RETRIEVAL_ARM
from rag.retrieval.lookup import load_lookup_keys
from rag.retrieval.pipeline import LEG_CANDIDATE_LIMIT, TOP_K


def _retrieval_config(settings_condenser_model: str, settings_condenser_provider: str) -> dict[str, object]:
    """What this run's retrieval and condensation stages actually ran (SPEC §12.11) — the
    ladder-winning arm's own pinned constants, plus the condenser's model/provider (SPEC
    §8.2: "a controlled constant"), since `RunHeader` has no dedicated field for it and this
    dict is the caller's free-form record of everything else that could move a score."""
    return {
        "retrieval_arm": GENERATION_RETRIEVAL_ARM,
        "embedder": EMBEDDER_MODEL_ID,
        "leg_candidate_limit": LEG_CANDIDATE_LIMIT,
        "top_k": TOP_K,
        "condenser_model": settings_condenser_model,
        "condenser_provider": settings_condenser_provider,
    }


def _print_summary(run: GenerationRun) -> None:
    n = len(run.items)
    state_correct = sum(1 for item in run.items if item.state_correct)
    citation_valid = sum(1 for item in run.items if item.citation_valid)
    scored = [item.citation_correctness for item in run.items if item.citation_correctness is not None]
    mean_correctness = sum(scored) / len(scored) if scored else None

    print(f"items: {n}")
    print(f"state accuracy: {state_correct}/{n}")
    print(f"citation validity: {citation_valid}/{n}")
    if mean_correctness is not None:
        print(f"citation correctness: {mean_correctness:.3f} (mean over {len(scored)} item(s) with gold articles)")
    else:
        print("citation correctness: no item in this run had gold articles")
    print(
        "  diagnostic pairing (SPEC §12.9): compare this against the ladder's article recall@4 "
        "(a different run, `eval/runs/rung*.json`) — recall high + correctness low is a "
        "generation failure, both low is a retrieval failure."
    )
    for name in ("faithfulness", "point_coverage"):
        judged = [getattr(item, name) for item in run.items if getattr(item, name) is not None]
        if judged:
            print(f"{name}: {sum(judged) / len(judged):.3f} (mean over {len(judged)} judged item(s))")
    if run.header.judge_model:
        print(f"judge: {run.header.judge_model} via {', '.join(run.header.judge_providers) or '(no call succeeded)'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--no-sync", action="store_true", help="skip syncing the golden set to Langfuse first"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help=(
            f"items to run at once (default {MAX_CONCURRENCY}); `dataset.run_experiment()` "
            "drops rather than retries an item that hits a transient upstream error (e.g. "
            "OpenRouter's shared per-model rate limit), so a lower value trades run time for "
            "a better shot at a clean, complete-item run"
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip faithfulness and point coverage — the three deterministic metrics only",
    )
    args = parser.parse_args(argv)

    settings = load_settings()  # also loads .env into the process environment
    # Built before any paid call, so a same-family or unconfigured judge fails up front.
    judge = None if args.no_judge else make_judge(settings)
    if not args.no_sync:
        print(f"syncing {GOLDEN_SET_PATH} -> Langfuse generation dataset ...")
        sync_generation_dataset(GOLDEN_SET_PATH)

    client = QdrantClient(settings.qdrant_url)
    from rag.ingest.embedder import embed_batch  # deferred: pulls in torch

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"generation-{settings.generation_model.split('/')[-1]}-{stamp}"
    try:
        lookup_keys = load_lookup_keys(client)
        run = run_generation_eval(
            client=client,
            embed=embed_batch,
            lookup_keys=lookup_keys,
            condense_fn=make_condense_fn(settings),
            generate_fn=make_generate_fn(settings),
            arm=settings.generation_model,
            run_id=run_id,
            repo_root=REPO_ROOT,
            golden_set_path=GOLDEN_SET_PATH,
            runs_dir=RUNS_DIR,
            generation_model=settings.generation_model,
            generation_provider=settings.generation_provider,
            retrieval_config=_retrieval_config(settings.condenser_model, settings.condenser_provider),
            retrieval_arm=GENERATION_RETRIEVAL_ARM,
            max_concurrency=args.max_concurrency,
            judge=judge,
        )
    except JudgeRunError as error:
        # The run is already written; its judged scores are incomplete, so no summary.
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print(f"written to eval/runs/{run_id}.json ({len(run.items)} item(s))")
    _print_summary(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
