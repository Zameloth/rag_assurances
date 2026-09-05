# ADR-0015 — Rung 1 is two per-register legs merged dense-only to top-8; the arm is selected by a code constant, not an env var

- **Status**: Accepted — 2026-09-05
- **Ticket**: [#28](https://github.com/Zameloth/rag_assurances/issues/28)
- **Spec**: [`SPEC.md` §9.2](../../SPEC.md#92-the-three-retrieval-paths), [`SPEC.md` §12.7](../../SPEC.md#127-the-pre-registered-rule), [`SPEC.md` §16.3](../../SPEC.md#163-configuration)

## Context

SPEC §12.7's ladder table names rung 1 in one phrase: *"naive baseline — single index, dense-only,
no expansion, no rerank, top-8."* That phrase predates ADR-0005, which already commits ingest to
**two** Qdrant collections behind two aliases (`fiches`, `articles`) — there is no literal single
collection for rung 1 to query. §12.7 is explicitly pre-registered *"before any rung runs"* to stop
a metric being picked after seeing numbers, so reconciling this wording gap belongs in a decision
record, not only in `rag.retrieval.pipeline`'s module docstring.

A second, smaller question rides along: issue #28's acceptance criteria ask that "the arm is
selectable by config, so rung 1 stays runnable after later rungs land." SPEC §16.3 fixes the
`.env` variable table exhaustively; adding an entry there for something that is an ablation choice,
not a secret or a deployment fact, would be the kind of scope SPEC §16.3 doesn't claim.

## Decision

**"Single index" reads as "one vector kind" (dense-only), not "one collection."** Rung 1 queries
both the `fiches` and `articles` aliases independently — SPEC §9.2's fiche leg / article leg
split, at their already-fixed depth of 20 each (`LEG_CANDIDATE_LIMIT`) — takes only the dense half
of each candidate's BGE-M3 embedding, merges the two pools by raw cosine score with no per-leg
weighting (that lands with #29), and slices to the top 8. No `<dc:source>` expansion, no rerank,
no register quota — those are later ladder rows.

**The retrieval arm is selected by a named code constant plus an injectable function parameter**
(`RETRIEVAL_ARMS: dict[str, RetrieveFn]`, `DEFAULT_RETRIEVAL_ARM`), mirroring `rag.ingest.pipeline`'s
existing `ARTICLES_ARM`/`FICHES_ARM` pattern, rather than a `RETRIEVAL_ARM` environment variable.

## Rationale

- **Two legs, not one merged collection**, because ADR-0005 already settled the storage question
  and nothing in #28 reopens it. Interpreting "single index" as "single collection" would demand
  re-merging `fiches` and `articles` — a storage-architecture change with no ticket behind it —
  just to satisfy one adjective in a table cell written before that architecture existed.
- **Per-leg depth stays at SPEC §9.2's `top-20`** rather than shrinking to something rung-1-sized,
  so the per-leg candidate pools in the fat object (`RetrievalResult.candidate_pools`) are
  comparable across every rung from the start — the eval harness (not yet built) reads candidate-
  depth recall the same way regardless of which rung produced the pool.
- **A code constant, not an env var, selects the arm** because SPEC §16.3's table is described as
  the configuration surface end to end ("every variable the pipeline reads appears here"); which
  retrieval arm runs is a code-level ablation choice exactly like `ARTICLES_ARM`/`FICHES_ARM`
  already are, not an environment-specific fact like `QDRANT_URL`.

## Consequences

- `rag.retrieval.pipeline.retrieve_rung1`'s per-leg limit (20) and final cap (8) are two different
  numbers on purpose; a future reader should not "fix" the per-leg limit down to 8 expecting it to
  match the ladder's rung-1 headline figure.
- Rung 2 (#29) reuses `search_leg` and `merge_candidates` unchanged and only changes how the merged
  pool is scored (per-leg weighted fusion instead of raw dense score) — this ADR's shape is what
  makes that a small diff rather than a rewrite.
- Adding a second retrieval arm to `RETRIEVAL_ARMS` is the whole integration surface for a future
  rung; no `Settings`/`.env.example` change is implied by that addition.
