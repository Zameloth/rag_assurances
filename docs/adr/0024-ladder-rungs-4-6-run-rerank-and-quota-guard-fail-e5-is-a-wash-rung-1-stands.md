# ADR-0024 — Ladder rungs 4-6 run: rerank and quota both guard-fail, e5 is a wash, rung 1 remains the deployed arm

- **Status**: Accepted
- **Ticket**: #41
- **Spec**: SPEC.md §9.4-§9.5, §12.6, §12.7, §14.4, §17.2, ADR-0011, ADR-0018, ADR-0019, ADR-0022, ADR-0023

## Context

Rungs 4-6 were built by #39/#41 (ADR-0018, ADR-0019, ADR-0022), following ADR-0023's rungs 1-3
run, but had not actually executed against the golden set. `scripts/run_ladder_rungs_4_6.py`
extends `scripts/run_ladder.py`'s discipline: rung 4 vs rung 3, rung 5 vs rung 4, rung 6 vs rung 5
— never against rung 1 directly — loading #40's already-committed rung 3 run rather than
re-running it, exactly as ADR-0023's own rung 2-vs-1/3-vs-2 pairing does.

Getting the reranker running for real (the first rung to do local CPU-bound cross-encoder
inference rather than call an API) surfaced two infrastructure bugs, fixed before this run:
`transformers` 5.x's removal of `prepare_for_model` broke BGE-M3's encode path, and the lazy
`@lru_cache` model loaders in `rag.ingest.embedder`/`e5_embedder`/`rag.retrieval.rerank` were not
safe against Langfuse's concurrent `task()` calls racing to construct the same cached model.

**A third bug surfaced running rung 5 for real and crashed the first attempt outright**:
`rag.retrieval.quota.assemble_quota`'s no-article marker — the explicit sentinel SPEC §9.5
requires when zero articles clear the relevance floor — is a synthetic `Candidate` with
`register is Register.ARTICLE` but no `legiarti_cid` in its payload (by design: it names no real
document). `rag.eval.retrieval_metrics.ranked_ids` indexed `candidate.payload["legiarti_cid"]`
unconditionally for every `Register.ARTICLE` candidate, so the first rung-5 item where the floor
rejected everything crashed the run (`KeyError: 'legiarti_cid'`). Worse than the crash itself:
had it not crashed, `_zero_articles` (`not any(candidate.register is Register.ARTICLE ...)`) would
have silently read the marker itself as "an article is present," making `zero_articles` — rung
5's own pre-registered primary metric — **structurally unable to ever read `True` on rung 5**,
and `floor_correct` wrong with it. Both functions were fixed to exclude the marker by id before
this run executed (`src/rag/eval/retrieval_metrics.py`, regression test in
`tests/eval/test_retrieval_metrics.py` built through the real `assemble_quota`, not a hand-rolled
stand-in). Rung 4's own committed run (no quota, so the marker never appears) predates the fix and
needed no re-run; rung 5 and rung 6 do not exist without it.

## Decision

No pipeline or scoring-rule change beyond the bug fix above. `run_ladder_rungs_4_6.py` ran rung 4
(reranker, incumbent collections), rung 5 (register quota + relevance floor, incumbent
collections), and rung 6 (e5-dense + M3-sparse, via a temporary alias flip to
`articles__e5-m3__c512__v1`/`fiches__e5-m3__c512__v1`, restored in a `finally`) — each compared
against the rung immediately below it, per ADR-0011's adoption rule: **adopt iff net discordant
pairs on the primary ≥ 4, and no other decision metric regresses by more than 1 net item.**

## Measured result

Ran on 2026-09-17 against the same 40-item single-turn retrieval working set ADR-0023 used
(33 of those 40 carry `gold_articles`). Rung 4's resident memory was measured earlier
(`scripts/measure_reranker_ram.py`, unaffected by the retrieval_metrics.py fix); rung 6's was
measured by `scripts/measure_e5_ram.py`.

| rung comparison | primary | net discordant | threshold | guard violations (net) | verdict |
|---|---|---|---|---|---|
| rung4 vs rung3 (+ reranker) | fiche_recall_at_4 (n=40) | **+14** (p≈0.000) | ≥ 4 ✓ | article_recall_at_4 (−7), zero_articles (−15) | **keep incumbent** |
| rung5 vs rung4 (quota + floor) | zero_articles (n=33) | −13 (p≈0.001) | ≥ 4 | fails on its own primary | **keep incumbent** |
| rung6 vs rung5 (e5-dense + M3-sparse) | fiche_recall_at_4 (n=40) | −1 (p=1.000) | ≥ 4 | none, but primary itself is flat | **keep incumbent** |

**Rung 4 repeats rung 3's shape exactly** (ADR-0023): a strong, significant win on its own primary
(14 of 40 items gain a previously-missed fiche in the top-4, span containment moves in lockstep)
purchased by degrading both article-side guards. This is the literal failure mode ADR-0019's own
context section names: *"a cross-encoder ranks by relevance to the query alone, and the query is
consumer French... naive top-8 can return eight fiche chunks and zero articles, intermittently."*
Reranking by relevance alone reshuffles fiche-favoring French consumer-phrasing hits above
articles that a raw fusion score had kept in the top 8, on 15 items net.

**Rung 5 does not fix what rung 4 broke — on this 33-item subset, it makes zero-article rate
measurably worse than rung 4's unguarded top-8** (13 net items regress from "at least one article
present" to "none," 1 improves the other way, sign-test p≈0.001 — the one comparison in this run
that clears conventional significance, in the wrong direction for adoption). The mechanism is the
floor, not the quota split itself: `ARTICLE_RELEVANCE_FLOOR = 0.5` is, by its own module
docstring, "a starting point for the ladder to calibrate, not a measured value" — the sigmoid
cross-encoder's own decision boundary, untuned against this corpus. At ~44 items the floor
rejects some genuinely-correct articles more often than the unguarded rung-4 top-8 happened to
drop them. `article_recall_at_candidate` stays flat (n=29, net 0) confirming the correct article
was still in the raw pool for these items — it is being rejected at the floor, not lost upstream.
This is exactly the outcome SPEC's own adoption rule anticipates and ADR-0011 names by name
("rungs 5 and 6 are *expected* to be inconclusive... a clean recorded outcome rather than an
argument") — it is a *sharper* inconclusive than a coin-flip null, not a different kind of result.

**Rung 6 (embedder A/B) is a wash.** Every decision metric nets to 0 or a one-to-two-item
regression (`gs-063` alone drives the `fiche_recall_at_4`/`_at_10` regressions; `gs-063` and
`gs-020` drive `span_containment_at_4`'s); nothing approaches the ≥4 threshold either direction.
e5-dense + M3-sparse neither helps nor hurts measurably at this N.

Full per-item scores are committed at `eval/runs/rung4-20260917T132523Z.json`,
`eval/runs/rung5-20260917T154321Z.json` and `eval/runs/rung6-20260917T154321Z.json` (rung 4's own
run predates the resumed rung5/6 timestamp — see Context — since it needed no re-run). All three
verdict computations are committed at `eval/runs/rung4-verdict-20260917T154321Z.json`,
`eval/runs/rung5-verdict-20260917T154321Z.json` and `eval/runs/rung6-verdict-20260917T154321Z.json`
(rung 4's verdict was recomputed against the same rung 3 baseline under the resumed run's
timestamp for this ADR's table; the numbers are unchanged from the pre-crash computation).

**Resident memory**: rung 4 (BGE-M3 + `bge-reranker-v2-m3`, fp32, both query-warmed) measures
**3581 MB**; rung 6 (BGE-M3 + `multilingual-e5-large-instruct` co-resident) measures **4574 MB**.
Both are below the ~4.5 GB (4500 MB) budget only in rung 4's case — rung 6 sits **~74 MB over**.
Moot for deployment (rung 6 did not clear the quality bar to begin with), but concrete rather than
estimated: had rung 6 won, SPEC §14.4's "prod runs the cheapest arm that fits" fallback would have
fired for real, not hypothetically.

## Consequences

- **No default retrieval arm changes.** `DEFAULT_RETRIEVAL_ARM` stays `"rung1"` — the same
  conclusion ADR-0023 already reached for rungs 2-3. Across all six rungs of the pre-registered
  ladder, none clears ADR-0011's adoption bar: production runs rung 1's naive two-leg dense-only
  top-8 arm, unchanged since ADR-0015.
- **Index-bearing vs. runtime, stated explicitly per SPEC §14.4's own split** (conflating them is
  "a mistake worth naming"): **index-bearing** — BGE-M3 dense+sparse, the two-collection
  `articles`/`fiches` layout, no enrichment flag change (ADR-0004/0005) — is unchanged; the e5 arm
  built by #39 stays reachable by alias flip but unused. **Runtime** — no reranker, no per-leg
  weight change beyond rung 1's even dense merge, no register quota, no relevance floor, no
  expansion — is likewise unchanged from rung 1's own config.
- **SPEC §17.2 ("whether prod can afford the ladder-winning reranker arm") resolves to moot, not
  answered**: there is no ladder-winning reranker arm to afford. The rung-6 RAM figure
  (4574 MB, ~74 MB over budget) is recorded for the record SPEC asks for, not acted on.
- **Rung 5's own result is now itself evidence for a follow-up, not a fixed conclusion**:
  `ARTICLE_RELEVANCE_FLOOR = 0.5` was always documented as an uncalibrated starting point: this
  run is the first data point suggesting it is currently mis-set (too strict) for this
  cross-encoder on this corpus, not proof the quota-and-floor *design* is wrong. Recalibrating the
  floor against a larger golden set is a natural next ticket; re-running rung 5 with a different
  floor value on the same ~44 items would burn the one clean measurement this ticket bought.
  This ADR does not open that ticket — it only preserves the evidence for it.
  - **Why the marker bug had to be fixed before drawing that conclusion at all**: without the
    fix, `zero_articles` would have been unable to distinguish rung 5's floor-rejection outcome
    from "an article present" in the first place — the "quota makes zero-article rate worse"
    finding above did not exist as a *measurable* claim until `ranked_ids`/`_zero_articles`
    learned to exclude the no-article marker.
- **Rung 4's article/zero-article guard failures are a concrete argument for pairing rerank with
  a quota**, even though rung 5 as pre-registered didn't clear that bar this run: a bare
  cross-encoder is not a safe drop-in replacement for the fused top-8 on its own.
