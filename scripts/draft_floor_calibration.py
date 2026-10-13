#!/usr/bin/env python3
"""Draft the floor-calibration candidates (#95, #84, ADR-0020).

    uv run python scripts/draft_floor_calibration.py [--n 45] [--seed 84]

Samples fiches no golden-set item uses (seeded, re-runnable), chains the annotation helper's
propose functions over each, runs `rag.eval.disagree` over every draft and writes:

- `eval/golden/floor-calibration.yaml` — golden schema, `history: []`, seed in the first line;
- `eval/golden/floor-calibration.review.yaml` — per item, the considered shortlist with each
  candidate's pick/reject justification, and the disagreement-pass finding.

Never imports `rag.retrieval` (ADR-0020). Needs `OPENROUTER_API_KEY` and `GENERATION_MODEL`;
costs several OpenRouter calls per fiche — run it by hand, not in CI.
"""

from __future__ import annotations

import argparse
import sys

from rag.config import load_settings
from rag.eval.corpus import load_articles
from rag.eval.disagree import DisagreementCallError
from rag.eval.floor_calibration_draft import draft_floor_calibration, write_drafts
from rag.eval.paths import GOLDEN_SET_PATH, REPO_ROOT
from rag.eval.schema import load_golden_set

FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"
CALIBRATION_GOLDEN_PATH = REPO_ROOT / "eval" / "golden" / "floor-calibration.yaml"
CALIBRATION_REVIEW_PATH = REPO_ROOT / "eval" / "golden" / "floor-calibration.review.yaml"

# ~45 on purpose (#95): oversampled, because sans-article items are rarer and the reviewer
# will reject some on the way to 20 + 10.
DEFAULT_N = 45
DEFAULT_SEED = 84


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="fiches to sample (default %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed (default %(default)s)")
    args = parser.parse_args(argv)

    try:
        run = draft_floor_calibration(
            fiches_dir=FICHES_DIR,
            articles=load_articles(ARTICLES_PATH),
            golden_items=load_golden_set(GOLDEN_SET_PATH),
            settings=load_settings(),
            n=args.n,
            seed=args.seed,
        )
    except DisagreementCallError as exc:
        print(f"disagreement pass aborted: {exc}", file=sys.stderr)
        return 1

    write_drafts(run, CALIBRATION_GOLDEN_PATH, CALIBRATION_REVIEW_PATH)

    sans = sum(1 for d in run.drafts if d.item.expected_state == "reponse_sans_article")
    print(f"seed {run.seed}: {len(run.drafts)} draft(s) ({sans} reponse_sans_article), {len(run.failures)} failed")
    for fiche_id, reason in sorted(run.failures.items()):
        print(f"  FAILED {fiche_id}: {reason}")

    print("\n=== Disagreements (independent pass vs drafted gold_articles) ===")
    if not run.disagreements:
        print("  none")
    for item_id, d in sorted(run.disagreements.items()):
        flag = "NEEDS REVIEW" if d.needs_review else "explained by <dc:source> pool"
        print(f"  {item_id} [{flag}]")
        print(f"    drafted gold_articles : {sorted(d.gold_articles) or '(empty)'}")
        print(f"    independent model pick: {sorted(d.model_picks) or '(empty)'}")
    print(f"\nwrote {CALIBRATION_GOLDEN_PATH}\nwrote {CALIBRATION_REVIEW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
