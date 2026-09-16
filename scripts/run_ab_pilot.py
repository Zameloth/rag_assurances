#!/usr/bin/env python3
"""Run one or both pre-ladder A/Bs end to end and record their verdicts (SPEC §12.8, #38).

For the named A/B (or both): sync the golden set to Langfuse, build the challenger arm if
it isn't already built, then run the *same fixed hybrid pipeline* (`pipeline_arm="rung2"` —
the earliest rung that actually uses both vector kinds, SPEC §9.3) against the incumbent
collection and again against the challenger collection, flipping the relevant stable alias
in between. Both runs are persisted (SPEC §12.11's `eval/runs/<run-id>.json`), then
`rag.eval.compare.compare_registered_runs` reads the two off `rag.eval.ladder_registry`'s
pre-registered primary for that A/B's `rung`.

**The stable alias is always restored to the incumbent arm before this script returns**,
success or failure (`try`/`finally`) — adopting a challenger is a deliberate, separate step
taken once a verdict is read (SPEC §12.8's own acceptance criterion: "the winning
enrichment flags become the incumbent config the ladder starts from"), never a side effect
of having measured it.

    uv run python scripts/run_ab_pilot.py article-breadcrumb
    uv run python scripts/run_ab_pilot.py fiche-header
    uv run python scripts/run_ab_pilot.py both
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from rag.config import load_settings
from rag.eval.compare import ComparisonReport, compare_registered_runs, write_comparison
from rag.eval.langfuse_sync import sync_retrieval_dataset
from rag.eval.retrieval_run import RetrievalRun
from rag.eval.run_experiment import run_retrieval_ladder
from rag.ingest.ab_arms import (
    ARTICLE_BREADCRUMB_ARM,
    FICHE_HEADER_ARM,
    build_article_breadcrumb_arm,
    build_fiche_header_arm,
)
from rag.ingest.arms import ARTICLES_ALIAS, FICHES_ALIAS, flip_alias
from rag.ingest.pipeline import ARTICLES_ARM, FICHES_ARM
from rag.ingest.upsert import EmbedFn
from rag.retrieval.lookup import load_lookup_keys

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SET_PATH = REPO_ROOT / "eval" / "golden" / "golden-set.yaml"
RUNS_DIR = REPO_ROOT / "eval" / "runs"

# SPEC §9.3 — the earliest ladder rung that fuses both vector kinds; rung 1 is dense-only
# and would make the "does breadcrumb noise hurt the sparse leg" half of §12.8's argument
# untestable by construction.
PIPELINE_ARM = "rung2"


@dataclass(frozen=True)
class AbSpec:
    name: str
    rung: str
    alias: str
    incumbent_arm: str
    challenger_arm: str
    build_challenger: Callable[[QdrantClient, EmbedFn], int]


AB_SPECS: dict[str, AbSpec] = {
    "article-breadcrumb": AbSpec(
        name="article-breadcrumb",
        rung="ab_article_breadcrumb",
        alias=ARTICLES_ALIAS,
        incumbent_arm=ARTICLES_ARM,
        challenger_arm=ARTICLE_BREADCRUMB_ARM,
        build_challenger=build_article_breadcrumb_arm,
    ),
    "fiche-header": AbSpec(
        name="fiche-header",
        rung="ab_fiche_header",
        alias=FICHES_ALIAS,
        incumbent_arm=FICHES_ARM,
        challenger_arm=FICHE_HEADER_ARM,
        build_challenger=build_fiche_header_arm,
    ),
}


def _run_one_side(
    *,
    client: QdrantClient,
    embed: EmbedFn,
    header_arm: str,
    spec: AbSpec,
    run_id: str,
) -> RetrievalRun:
    lookup_keys = load_lookup_keys(client)
    return run_retrieval_ladder(
        client=client,
        embed=embed,
        lookup_keys=lookup_keys,
        arm=header_arm,
        rung=spec.rung,
        pipeline_arm=PIPELINE_ARM,
        run_id=run_id,
        repo_root=REPO_ROOT,
        golden_set_path=GOLDEN_SET_PATH,
        runs_dir=RUNS_DIR,
        retrieval_config={
            "pipeline_arm": PIPELINE_ARM,
            "collection_arm": spec.challenger_arm if header_arm == "challenger" else spec.incumbent_arm,
        },
    )


def run_ab(client: QdrantClient, embed: EmbedFn, spec: AbSpec, *, stamp: str) -> ComparisonReport:
    print(f"=== {spec.name} ===")
    print(f"building challenger arm {spec.challenger_arm!r} ...")
    written = spec.build_challenger(client, embed)
    print(f"  {written} point(s) written")

    try:
        flip_alias(client, spec.alias, spec.incumbent_arm)
        print(f"[{spec.alias}] -> {spec.incumbent_arm} (incumbent) — running ...")
        incumbent_run = _run_one_side(
            client=client,
            embed=embed,
            header_arm="incumbent",
            spec=spec,
            run_id=f"{spec.rung}-incumbent-{stamp}",
        )

        flip_alias(client, spec.alias, spec.challenger_arm)
        print(f"[{spec.alias}] -> {spec.challenger_arm} (challenger) — running ...")
        challenger_run = _run_one_side(
            client=client,
            embed=embed,
            header_arm="challenger",
            spec=spec,
            run_id=f"{spec.rung}-challenger-{stamp}",
        )
    finally:
        flip_alias(client, spec.alias, spec.incumbent_arm)
        print(f"[{spec.alias}] restored -> {spec.incumbent_arm}")

    report = compare_registered_runs(incumbent_run, challenger_run)
    verdict_path = RUNS_DIR / f"{spec.rung}-verdict-{stamp}.json"
    write_comparison(report, verdict_path)
    _print_report(report)
    print(f"verdict written to {verdict_path}\n")
    return report


def _print_report(report: ComparisonReport) -> None:
    print(f"primary: {report.primary_metric}")
    for metric, comparison in report.metrics.items():
        print(
            f"  {metric:<28} n={comparison.n_paired:>2}  improved={comparison.improved:>2}  "
            f"regressed={comparison.regressed:>2}  net={comparison.net_discordant:+d}"
        )
    verdict = report.verdict
    outcome = "ADOPT" if verdict.adopt else "KEEP INCUMBENT"
    print(f"VERDICT: {outcome} (net on primary = {verdict.net_discordant_primary:+d})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=["article-breadcrumb", "fiche-header", "both"])
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
    names = list(AB_SPECS) if args.which == "both" else [args.which]
    try:
        for name in names:
            run_ab(client, embed_batch, AB_SPECS[name], stamp=stamp)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
