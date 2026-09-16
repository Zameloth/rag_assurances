# ADR-0023 — Ladder rungs 1-3 run: rung 3 clears article recall but trips the fiche and floor guards

- **Status**: Accepted
- **Ticket**: #40
- **Spec**: SPEC.md §9.1-§9.3, §12.6, §12.7, ADR-0011, ADR-0015, ADR-0016, ADR-0017

## Context

Rungs 1-3 were built and made ablatable by #29/#30 (ADR-0015/0016/0017), but none had actually run
against the golden set and been judged by the pre-registered adoption rule — #37/ADR-0011's table
only committed *which* metric each rung would be judged on, before any rung ran. #38 exercised the
harness on the two pre-ladder A/Bs; this ticket is the first run of the ladder itself.

## Decision

`scripts/run_ladder.py` drives all three rungs end to end: sync the golden set, then run
`rung1`/`rung2`/`rung3` each exactly once against the same incumbent `articles`/`fiches`
collections — no alias flip, unlike `run_ab_pilot.py`'s challenger collections, since every rung
here reads the same index and only the pipeline changes. Each run is persisted as its own
`eval/runs/<rung>-<stamp>.json`. **Each rung is compared against the rung immediately below it**
(rung2 vs rung1, rung3 vs rung2), never against rung1 alone — that mirrors "each rung changes
exactly one variable from the rung below it" (#40's own acceptance criterion), and rung1 has no
pre-registered primary to compare against in the first place (`ladder_registry.primary_metric_for`
raises for it — SPEC §12.7: "reference floor, not a comparison"). Both verdicts are computed by
`rag.eval.compare.compare_runs` with the primary resolved from `ladder_registry.primary_metric_for`,
never typed into the script by hand.

## Measured result

Ran on 2026-09-16 against the full committed corpus and the current 60-item golden set's 40-item
single-turn (`history == []`) retrieval working set.

| rung comparison | primary | net discordant | threshold | guard violations | verdict |
|---|---|---|---|---|---|
| rung2 vs rung1 (+ hybrid sparse leg) | article_recall_at_4 (n=33) | −3 (p=0.25) | ≥ 4 | zero_articles (−14 net) | keep incumbent |
| rung3 vs rung2 (+ `<dc:source>` expansion) | article_recall_at_4 (n=33) | **+10** (p≈0.002) | ≥ 4 ✓ | fiche_recall_at_4 (−13 net), floor_correct (−7 net) | **keep incumbent** |

Rung 2's hybrid sparse leg move is inconclusive on its own primary and additionally regresses
zero-article rate sharply (guard, though not the reason it fails — the primary already misses
threshold on its own).

**Rung 3 clears the adoption bar on its own primary outright**: net +10 discordant pairs on
article_recall@4 (10 improved, 0 regressed, sign-test p≈0.002), and zero_articles improves on 29 of
33 paired items. This is the confirmation the headline experiment set out to get — a curated
`<dc:source>` join surfaces gold articles that embedding similarity alone misses.

**It is still not adopted.** `compare_runs` checks every decision metric, not only the SPEC table's
named guard (fiche_recall_at_4): fiche_recall_at_4 regresses net −13 (13 of 40 paired items lose a
previously-correct fiche), and floor_correct — the 8 `reponse_sans_article` items — regresses on
all 7 paired items where it applies. Both trace to the same mechanical cause: rung 3 adds a third
candidate pool (expansion) feeding the same shared top-8 cap (`rank_candidates`, SPEC §9.2) rungs
1-3 all still use. Nothing in the current pipeline reserves slots per register, so every
expansion-sourced article that wins a slot does so by displacing a fiche or, on
`reponse_sans_article` items, by filling a slot expansion should never have touched. This is exactly
the intermittent failure SPEC §9.5/ADR-0019's register quota (rung 5) exists to design out — rung 3
answers "does the curated join have signal at all" (yes); "does the current cap allocate that signal
safely" (not yet) is a different question, and not this ticket's to answer.

Per SPEC §12.7 rule 3, inconclusive/guard-violated always resolves to *no change*. Full per-item
scores and both verdict computations are committed at `eval/runs/rung{1,2,3}-20260916T181814Z.json`
and `eval/runs/rung{2,3}-verdict-20260916T181814Z.json`.

## Consequences

- No default retrieval arm changes as a result of this ticket — `DEFAULT_RETRIEVAL_ARM` stays
  `"rung1"`; rungs 1-3 remain reachable by name through `RETRIEVAL_ARMS`.
- Rung 3's guard failures (fiche_recall_at_4, floor_correct) are exactly the two symptoms rung 5's
  register quota is designed to fix by reserving fiche/article slots separately and rejecting
  under-floor candidates rather than letting them silently win a shared slot — a concrete
  prediction rung 5's own run can be checked against.
- Rung 2's zero_articles regression is a smaller instance of the same shared-top-8-cap mechanism:
  hybridizing the article leg's fusion surfaces more borderline candidates that can crowd out clean
  top-4 picks on some items, before expansion makes the effect much larger at rung 3.
