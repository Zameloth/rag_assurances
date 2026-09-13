#!/usr/bin/env python3
"""Run the disagreement-detector pass over the finished golden set (ADR-0010, ADR-0020, #34).

**A dev tool, run by hand**, and a **separate, later** pass — never during annotation
(ADR-0020: proposal-first assistance would make it "redundant with itself, not
independent"). For every retrieval-bearing item, a model independently picks gold articles
from only its `gold_fiches`' `<dc:source>` sections, blind to the annotator's saved
`gold_articles`. Disagreements are printed for human re-review; **this script never writes
to `eval/golden/golden-set.yaml`** — it has no authority to change a label (ADR-0010).

Also runs the structural cross-check for ADR-0010's "at least one item's gold article lies
outside the sections its fiche cites" acceptance criterion — no LLM needed for that part.

    uv run python scripts/disagreement_check.py [path/to/golden-set.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path

from rag.config import load_settings
from rag.eval.corpus import load_articles
from rag.eval.disagree import (
    DisagreementCallError,
    find_outside_section_items,
    run_disagreement_pass,
)
from rag.eval.schema import load_golden_set

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_SET = REPO_ROOT / "eval" / "golden" / "golden-set.yaml"
FICHES_DIR = REPO_ROOT / "data" / "corpus" / "fiches"
ARTICLES_PATH = REPO_ROOT / "data" / "corpus" / "articles.jsonl"


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else DEFAULT_GOLDEN_SET
    items = load_golden_set(path)
    articles = load_articles(ARTICLES_PATH)

    print("=== Cross-check: gold article outside its own <dc:source> sections ===")
    outside = find_outside_section_items(items, FICHES_DIR, articles)
    if outside:
        for item_id, cids in sorted(outside.items()):
            print(f"  {item_id}: {sorted(cids)}")
        print(
            f"  -> criterion met ({len(outside)} item(s) evidence the reading aid was used as a reading aid)"
        )
    else:
        print("  NONE — ADR-0010's acceptance criterion requires at least one such item.")

    print(f"\n=== Disagreement-detector pass ({len(items)} item(s)) ===")
    settings = load_settings()
    try:
        disagreements = run_disagreement_pass(items, FICHES_DIR, articles, settings)
    except DisagreementCallError as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 1

    if not disagreements:
        print(
            "  no disagreements — every item's independent pick matched its gold_articles exactly."
        )
        return 0

    needing_review = 0
    for d in disagreements:
        print(f"\n  {d.item_id}")
        print(f"    gold_articles     : {sorted(d.gold_articles) or '(empty)'}")
        print(f"    model independent : {sorted(d.model_picks) or '(empty)'}")
        expected = d.explained_by_outside_section
        if expected:
            print(
                f"    outside <dc:source> sections (expected by design, ADR-0010): {sorted(expected)}"
            )
        if d.needs_review:
            needing_review += 1
            genuine_missing = d.only_in_gold - expected
            if genuine_missing:
                print(
                    f"    NEEDS REVIEW — annotator picked it, model saw it in-section but didn't: {sorted(genuine_missing)}"
                )
            if d.only_in_model:
                print(
                    f"    NEEDS REVIEW — model picked, annotator didn't: {sorted(d.only_in_model)}"
                )

    print(
        f"\n{len(disagreements)} item(s) flagged, {needing_review} needing re-review by hand — "
        "this pass has no authority to change a label."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
