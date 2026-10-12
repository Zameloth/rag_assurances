# ADR-0027 — Floor recalibration: a pre-registered sweep on a separate, LLM-drafted calibration set, then a held-out rung 4 vs rung 5 re-run

- **Status**: Proposed
- **Ticket**: #84 (parent), #96 (this pre-registration and its tooling)
- **Spec**: SPEC.md §9.5, §12.7, ADR-0010, ADR-0011, ADR-0019, ADR-0020, ADR-0024

## Context

ADR-0024 recorded rung 5 (register quota + relevance floor) making the zero-article rate
measurably *worse* than rung 4's unguarded top-8 (net −13, p≈0.001) while
`article_recall_at_candidate` stayed flat: correct articles reach the pool and are rejected at
the floor. `ARTICLE_RELEVANCE_FLOOR = 0.5` was always documented as the cross-encoder sigmoid's
midpoint, an uncalibrated starting point (`rag.retrieval.quota`, ADR-0019). ADR-0024 named
recalibrating it as a natural next ticket and warned that re-running rung 5 with other floor
values on the same golden-set items would burn the one clean measurement.

This ADR is the pre-registration for that ticket (#84). It is committed **before any sweep
runs**, for the same reason ADR-0011 pre-registers metrics: a selection rule chosen after
seeing the sweep is post-hoc metric selection.

Design is unchanged. SPEC §9.5's quota-and-floor design, `DEFAULT_RETRIEVAL_ARM = "rung1"` and
production are untouched; only the threshold value is in question.

## Decision

### 1. Calibrate on new data, test on the golden set

The floor is tuned on a **floor-calibration set** (`eval/golden/floor-calibration.yaml`): golden
schema, single-turn only, **20 article items + 10 `reponse_sans_article` items**. The
`reponse_sans_article` items are deliberately over-weighted (10 of 30, against 8 of ~44 in the
golden set) because they are the only signal that pushes the floor *up*; article items alone
can only ever argue for a lower floor. No `gold_fiches` entry is shared with `golden-set.yaml`
(enforced by the validator), so the golden set remains an untouched held-out test.

### 2. The sweep

- **Grid**: `relevance_floor` from 0.00 to 0.50 inclusive, step 0.025 (21 values).
- **Every other setting at rung-5 values**: fiche/article quota 4/4, fusion weights, expansion
  depth and cap, reranker model and backend, and `EXPANSION_PREFERENCE_MARGIN = 0.1`.
- **`EXPANSION_PREFERENCE_MARGIN` is out of scope.** It only reorders articles that are already
  above the floor, so it cannot explain a zero-article regression: whatever the margin, an
  article either clears the floor or does not. Sweeping it would add a second free parameter
  to a 30-item set for no gain on the failure under investigation.
- **Mechanics**: the rung-5 pipeline runs up to and including rerank **once** per item; each
  item's scored article pool is cached (ids, scores, provenance, no text); `assemble_quota` is
  then evaluated offline at every grid value. The sweep output carries every pool and every
  per-floor score, so the pick can be recomputed without the reranker.

### 3. Selection rule

For each floor, the **score** is the number of correct article-side decisions:

- an article item counts when at least one of its `gold_articles` is in the quota's article
  slots at that floor; plus
- a `reponse_sans_article` item counts when the floor rejects every article (the no-article
  marker takes the slots, `floor_met` is false).

Items that take the short-circuit path (SPEC §9.1) have no pool to score and are excluded from
every floor's score and listed in the output.

The chosen floor is the one with the **highest score; ties go to the higher floor** (SPEC §9.5,
"reject rather than pad"). The tie-break matters: with few items several adjacent floors will
tie, and the stricter one is the one the design prefers.

**A chosen floor of 0.0 means "no workable floor."** At 0.0 nothing is ever rejected, so the
`reponse_sans_article` items can only be wrong; a best of 0.0 means no threshold on this
cross-encoder separates the two populations better than not rejecting at all. That is recorded
as the result, and **the ticket stops**: no held-out re-run happens.

### 4. Held-out test

If the chosen floor is above 0.0, rung 4 and rung 5 (at the chosen floor) are re-run **together**
on the golden set's working set, under ADR-0011 **unchanged**: primary `zero_articles` (rung 5's
registered primary), adopt iff net discordant pairs ≥ 4, and no other decision metric
regressing by more than 1 net item. Rung 5 is compared to rung 4, as in ADR-0024, not to rung 1.

- **Pass**: this ADR is marked Accepted and a follow-up ticket compares rung 4 + rung 5 against
  rung 1 (the deployed arm). Production does not change in this ticket.
- **Fail, or best floor 0.0**: record the result, close the ticket, and name design-change
  options without opening them.

### 5. Post-hoc caveat

We retest *because* ADR-0024's result was bad. The golden set has already been seen, once, by
this exact comparison, and the floor is being re-examined only after it disappointed. The new
calibration set and the held-out structure limit the damage (no floor value is chosen with
golden-set data), but this is a second look at a failed result, not a fresh pre-registered
hypothesis, and any pass is weaker evidence than ADR-0023/ADR-0024's first-look verdicts. It is
recorded with the result either way.

### 6. Deviation from ADR-0010

ADR-0010 makes the golden set hand-authored, with AI assistance limited to proposals the
annotator accepts. The calibration set deviates on purpose: its items are **LLM-drafted and
human-validated** (confirmed or discarded, not rewritten from scratch). Calibration labels only
need to be roughly right: they steer a single threshold, and a mislabel moves one item's vote
at a handful of grid points. The hand-authored golden set stays the held-out judge, which is
where label quality matters. Drafting uses the lexical-shortlist helpers and never
`rag.retrieval` (ADR-0020), `disagree.py` serves as a second opinion, and the ladder scripts
never read `floor-calibration.yaml`.

## Consequences

- The calibration file, its validator rules (single-turn, no `gold_fiches` overlap with the
  golden set) and the sweep script land with #96, tested on fixtures only. No real sweep runs
  until this ADR is merged.
- The sweep's output file (`eval/runs/floor-sweep-<ts>.json`) is committed with the result, as
  every other run is (SPEC §12.11).
- Thirty items is small. The sweep reads a plateau of tied floors as "any of these"; the
  tie-break resolves it, it does not make it a precise estimate.
