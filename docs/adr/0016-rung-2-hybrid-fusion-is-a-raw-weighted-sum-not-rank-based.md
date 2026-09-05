# ADR-0016 — Rung 2's client-side fusion is a raw weighted sum of dense/sparse scores, with 2:1-leaning default weights

- **Status**: Accepted — 2026-09-05
- **Ticket**: [#29](https://github.com/Zameloth/rag_assurances/issues/29)
- **Spec**: [`SPEC.md` §9.2](../../SPEC.md#92-the-three-retrieval-paths), [`SPEC.md` §9.3](../../SPEC.md#93-hybrid--bge-m3-learned-sparse-weighted-per-leg), [ADR-0006](0006-three-path-hybrid-retrieval-with-editorial-expansion.md), [ADR-0015](0015-rung-1-two-legs-dense-only-top-8-arm-by-code-constant.md)

## Context

SPEC §9.3 and ADR-0006 already settle the shape — dense+sparse on both legs, weighted per leg,
fused client-side because Qdrant's RRF has no weight knob — and ADR-0015 pre-declares how rung 2
should land: `search_leg` and `merge_candidates` reused unchanged, only "how the merged pool is
scored" changes. Two things neither document pins down:

1. **What "weighted" means arithmetically.** A rank-based method (weighted RRF: `Σ weight / (k +
   rank)`) and a score-based method (weighted sum of the two raw scores) both honour "per-leg
   weights, client-side, not Qdrant's RRF" — they are not the same computation, and SPEC §9.3's
   regime table (MIRACL/MLDR nDCG numbers) doesn't distinguish between them.
2. **What the default weight values are.** SPEC §9.3 says "dense-leaning" / "sparse-leaning"
   qualitatively; no ticket assigns numbers.

## Decision

**`fuse_scores` (`rag.retrieval.fusion`) is a weighted sum of raw scores: `weight.dense *
dense_score + weight.sparse * sparse_score`, zero contribution from a side that didn't return the
id at all.** No rank transform, no min-max or z-score normalisation of either side. This is the
same shape `BGEM3FlagModel`'s own reference `compute_score` uses for its dense/sparse/colbert
blend — the model that produced both scores already ships an opinion on how to combine them
without normalising first, and cosine similarity (dense) and a learned-lexical-weight dot product
(sparse) are already comparable magnitudes because both come from the one BGE-M3 forward pass
this project standardised on (ADR-0004), not from two different models with different score
distributions.

**Default weights are 2:1** — `FICHE_LEG_WEIGHTS = LegWeights(dense=2/3, sparse=1/3)`,
`ARTICLE_LEG_WEIGHTS = LegWeights(dense=1/3, sparse=2/3)` — mirroring `compute_score`'s own
dense:sparse default ratio (colbert dropped, proportionally renormalised) rather than a round
number invented for this ticket. **This is a starting point for the ladder to tune, not a value
measured against this corpus** — SPEC's own design convention ("measure before deciding") applies
to claims of fact, not to a knob the eval harness exists to sweep later.

**`retrieve_rung2` takes `fiche_weights`/`article_weights` as keyword-only parameters,
defaulting to the two constants.** `hybrid_leg` already took weights as a parameter rather than
reading a module global; without threading that same override up through `retrieve_rung2`, the
only way to change what rung 2 actually runs with would be editing `fusion.py`'s constants —
config in name only. This mirrors `RETRIEVAL_ARMS`/`arm` (ADR-0015): a caller can override which
rung runs *and* what a given rung's tunable runs with, both without a code edit.

**Two new `query_points` calls per leg, not `prefetch`+server-side `FusionQuery`.** `search_leg`
(dense) is joined by a sibling `search_leg_sparse` in `legs.py`; `hybrid_leg` in the new
`fusion.py` calls both and fuses the results itself. Qdrant's `Prefetch`/`FusionQuery` API could
run both sub-queries in one round trip, but its only built-in fusion method is RRF — exactly what
SPEC §9.3 rules out — so using it would still require pulling both ranked lists back out and
re-fusing them client-side, at which point the one-round-trip savings buys nothing this design
needs.

## Rationale

- **Score-based over rank-based** because rank-based fusion (RRF-with-weights) throws away the
  magnitude information a weighted-sum keeps — two hits both ranked #1 in their leg but with very
  different confidence collapse to the same rank-based contribution, which is a worse fit for "how
  much do I trust dense vs. sparse *here*" than a method that can reflect a landslide dense hit
  outscoring a marginal sparse one.
- **No normalisation step** because introducing one (min-max over the pool, softmax, z-score)
  would be a free variable this ticket has no evidence to set, and BGE-M3's own reference scoring
  already answers "how do you combine this model's two score kinds" without one.
- **2:1 over a fresh round number** because inventing a ratio with no grounding is worse than
  reusing the ratio the embedding model's own authors picked for combining its two score kinds,
  even though neither is "the" answer — both are starting points, and one of the two comes from
  people who measured against the model directly.

## Consequences

- `LegWeights` doesn't need to sum to 1 (`fuse_scores` never normalises against the sum) — a
  future reader tuning weights should not "fix" them to look like a probability distribution.
- Adding a normalisation step later (if the ladder's rung-2-vs-rung-1 numbers suggest raw dot
  products are dominating unfairly) is a change to `fuse_scores` alone; `hybrid_leg`, `legs.py` and
  `pipeline.retrieve_rung2` would not need to change shape.
- `search_leg_sparse` and `search_leg` share a private `_search` helper in `legs.py` — a
  refactor of `search_leg`'s body, not its signature or behaviour, so it stays "reused unchanged"
  in the sense ADR-0015 meant (external contract), while avoiding duplicating the
  `Candidate`-construction logic ADR-0015's version already had.
