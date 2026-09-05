# ADR-0017 — Rung 3's expansion pool is additive, positional over the ranked fiche pool, and a filtered vector search

- **Status**: Accepted — 2026-09-05
- **Ticket**: [#30](https://github.com/Zameloth/rag_assurances/issues/30)
- **Spec**: [`SPEC.md` §9.2](../../SPEC.md#92-the-three-retrieval-paths), [ADR-0006](0006-three-path-hybrid-retrieval-with-editorial-expansion.md), [ADR-0016](0016-rung-2-hybrid-fusion-is-a-raw-weighted-sum-not-rank-based.md)

## Context

SPEC §9.2 and ADR-0006 already settle the shape — top-3 fiches expand into the `LEGISCTA`
sections they cite, `MatchAny` on the indexed `section_id`, a filtered vector search (never a
`scroll`) fills a cap of 40 — and name why: the cap exists to bound a candidate set, not to
select it, so ordering that set by point id (a `scroll`'s only order) would make the rung
partly measure the hash function instead of the editorial join. Three things neither document
pins down:

1. **Which fiches are "top 3."** The fiche leg (`hybrid_leg`, #29) already returns a pool sorted
   by fused score; nothing forces expansion to re-rank it.
2. **What "joins the fused pool" means for the merge.** An article reachable by both the article
   leg and expansion needs both provenance values recorded (SPEC §7.5) regardless of which pool
   the merge sees first.
3. **Where `depth`/`cap` live.** CONTEXT.md already classes the expansion cap as "runtime... app
   config", the same bucket ADR-0016 put per-leg weights in.

## Decision

**"Top 3" is positional over the already-ranked fiche pool — `top_fiches` takes the first
`depth` elements of `fiche_pool`, no second sort.** `hybrid_leg` (or `search_leg` at rung 1) has
already sorted by score descending; re-ranking here would silently diverge from what "top" meant
one line up and double the sorting logic for no behavioural difference.

**`section_ids_from_fiches` reads `section_ids` off each fiche chunk's payload and unions them,
deduped, order-preserving.** SPEC §7.1: `section_ids` is read by expansion, never filtered on —
this is that read, and nothing else in the codebase reads it. A fiche chunk with no
`section_ids` (or an empty list) contributes nothing rather than erroring; a fiche genuinely
out of the Code des assurances' scope is not a bug (SPEC §1.2).

**`expand_by_section` is one `query_points` call: `query_filter` `MatchAny` on `section_id`,
`query` the turn's own dense vector, `using="dense"`, capped at `cap`.** Empty `section_ids`
(no fiche in the top-`depth` window carries any) short-circuits to an empty pool rather than
issuing a filterless query — a filter-less `query_points` call would silently become an
unfiltered top-`cap` dense search, exactly the "gate demoted to nothing" failure SPEC §9.2's
gate-vs-sort-order distinction rules out.

**Expansion is additive, not a replacement leg.** `retrieve_rung3` calls `hybrid_leg` for both
fiche and article legs exactly as `retrieve_rung2` does, then calls `expand` as a third,
independent step, then merges all three pools with the existing `merge_candidates` (unchanged
since #28) rather than a rung-3-specific merge function. `merge_candidates` already unions
provenance sets on a repeated id regardless of which pool is passed first — it was written
with this ticket in mind (`candidates.py`'s own docstring: "this only matters once expansion
lands") — so an article reached by both the article leg and expansion carries
`{SEARCH, EXPANSION}` (or `{EXPANSION}` alone) no matter the argument order `merge_candidates`
is called with.

**`depth`/`cap` are keyword-only parameters on both `expand` and `retrieve_rung3`, defaulting to
`EXPANSION_FICHE_DEPTH`/`EXPANSION_CAP` (3/40, SPEC §9.2's own numbers).** The same shape
ADR-0016 gave `fiche_weights`/`article_weights`: a named constant a caller can override without
editing `expansion.py`, so the eval ladder can ablate fiche depth and the expansion cap without
a code change.

## Rationale

- **Positional "top 3" over a re-sort** because the fiche pool's score is already the fused
  hybrid score (#29) at rung 3 — re-deriving an order here would be a second, redundant
  definition of "most relevant fiche" that could drift from the first.
- **A dedicated `expand_by_section`/`expand` pair over folding expansion into `hybrid_leg`**
  because expansion is a different shape of query (filtered dense-only, one collection, driven
  by *another leg's* output) — forcing it through `hybrid_leg`'s dense+sparse+fuse contract would
  mean threading an unused sparse vector and a filter parameter `hybrid_leg`'s two current call
  sites (fiche leg, article leg) never need.
- **Reusing `merge_candidates` over a new merge function** because SPEC §7.5's union rule is not
  rung-specific — it is "how any two provenances on the same id combine", and #28 already built it
  that way; a second implementation would be two places to keep in sync with the same rule.

## Consequences

- `RetrievalResult.candidate_pools` gains a third key (`EXPANSION_POOL = "expansion"`) at rung 3;
  rungs 1 and 2 are untouched, so their `candidate_pools` shape stays exactly `{fiche_leg,
  article_leg}`.
- The expansion pool's scores (dense cosine, filtered) are not on the same scale as the article
  leg's (hybrid-fused dense+sparse) or as the fiche leg's — `rank_candidates` still sorts the
  merged pool by raw score regardless, the same pre-rerank posture rungs 1–2 already have.
  Making these scores comparable is rung 4's problem (reranking), not this ticket's.
- A fiche with no `section_ids` inside the top-`depth` window (or a fiche pool shorter than
  `depth`) degrades `expand` to an empty pool, never an error — the same "scope gap is not a bug"
  posture `rag.ingest.fiches.in_scope` already takes for ingest-time out-of-scope fiches.
