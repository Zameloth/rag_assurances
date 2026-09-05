# ADR-0018 — Rung 4's rerank replaces the merged pool's score outright, and its two RAM levers are parameters, not code

- **Status**: Accepted — 2026-09-05
- **Ticket**: [#31](https://github.com/Zameloth/rag_assurances/issues/31)
- **Spec**: [`SPEC.md` §9.4](../../SPEC.md#94-reranking), [`SPEC.md` §14.4](../../SPEC.md#144-ram--the-constraint-is-concurrency-not-capacity), [ADR-0017](0017-rung-3-expansion-is-an-additive-filtered-search-not-a-scroll.md)

## Context

SPEC §9.4 and §14.4 already settle the shape — `bge-reranker-v2-m3` over the fused pool, ablatable,
with `gte-multilingual-reranker-base` and int8 ONNX quantisation as the two RAM levers if the full
reranker doesn't fit prod's ~4.5 GB budget alongside BGE-M3 — and ADR-0017 flagged the open question
this ticket has to close: unifying the merged pool's score scale across three incomparable sources
(hybrid-fused dense+sparse on both legs, filtered dense cosine on the expansion pool) is "rung 4's
problem." Three things neither document pins down:

1. **Does rerank replace scores or add a parallel ranking?** SPEC §9.4 never says "sort by
   cross-encoder score" versus "keep the fused score and add a rerank score alongside it."
2. **How the two RAM levers actually reach the running arm.** SPEC §14.4 calls them levers "reachable
   at deploy time," but doesn't say through what mechanism — an environment variable, or a parameter,
   the way `fiche_weights`/`expansion_depth` already work (#29, #30).
3. **Where the hand-wrapped Langfuse span lives.** SPEC §11.1 requires one (the reranker isn't a
   LangChain component), but the #31 issue comment carves LangChain/Langfuse code out as paired work
   with @Zameloth, the same split #28 already drew around `langchain_retriever.py`.

## Decision

**`rerank()` (`rag.retrieval.rerank`) replaces every candidate's `score` outright and returns the pool
reordered by it, descending.** No parallel "rerank_score" field, no blending with the incoming fused
score. A cross-encoder's whole reason to exist here is that the incoming scores aren't comparable
across the pool's three sources (ADR-0017) — averaging a cross-encoder score with a scale it was
brought in to replace would just reintroduce the problem in a diluted form.

**`retrieve_rung4` is `retrieve_rung3` plus one step.** Short-circuit, both hybrid legs, expansion and
`merge_candidates` are `retrieve_rung3`'s own calls, byte-for-byte; rerank runs once, on the merged
pool, before `rank_candidates`' top-K cap. `RetrievalResult.candidate_pools` keeps rung 3's exact
shape (`fiche_leg`/`article_leg`/`expansion`) — reranking has nothing to do with which leg or
expansion found a candidate, only how the whole merged set is finally ordered, so it is not a fourth
candidate pool.

**The two RAM levers are `model_id`/`backend` keyword parameters on `rerank()`, forwarded as
`reranker_model`/`reranker_backend` on `retrieve_rung4`** — the same shape ADR-0016/ADR-0017 already
gave `fiche_weights`/`article_weights`/`expansion_depth`/`expansion_cap`. This ticket's acceptance
criterion ("reachable by config, not a code change") is satisfied the way this codebase already
defines that phrase for every other runtime knob: a caller overrides the default without editing
`rerank.py`. No `.env`/`Settings` variable was added for either lever — `fiche_weights` and
`expansion_depth` are already classed as "runtime... app config" in CONTEXT.md and neither was wired
to an environment variable either; inventing that wiring now, for a config-reading app layer that
does not exist yet (`src/rag/app/` is still empty), would be scope this ticket doesn't need. Whichever
app eventually reads `RERANKER_MODEL`/`RERANKER_BACKEND`-style settings passes them straight through as
these same two parameters, unchanged.

**`RerankerBackend` is a two-member enum, `FP32`/`ONNX_INT8`.** `FP32` loads whichever `model_id`
through `FlagEmbedding.FlagAutoReranker.from_finetuned` — both SPEC-named models
(`bge-reranker-v2-m3`, `gte-multilingual-reranker-base`) are already registered in FlagEmbedding's own
`AUTO_RERANKER_MAPPING`, so this needed no new dependency. `ONNX_INT8` exports and dynamically
quantises whichever `model_id` **locally**, once, caching the result under
`data/raw/hf_cache/onnx-int8/<model_id>` — rather than depending on a pre-quantised file published
under some assumed name on the hub, which would make the lever hostage to a third party's export
choices. This needs `optimum[onnxruntime]`, gated behind a new `rerank-onnx` dependency group (not
installed by default) — the same posture the `fetch` group already takes for `pyarrow`/`httpx`, a
lever exercised rarely enough that it shouldn't cost every `uv sync`.

**The hand-wrapped Langfuse span is not implemented in this ticket.** SPEC §11.1's requirement stands,
but per the #31 issue comment it is paired work with @Zameloth, landing in `langchain_retriever.py`
(or wherever that boundary grows) — the same split #28 drew around the `BaseRetriever`/
`CallbackHandler` wiring itself. `rerank.py` and `pipeline.py` import neither `langfuse` nor
`langchain`; both files carry a docstring note pointing at this gap rather than silently leaving it
undiscoverable.

## Rationale

- **Replace, not blend,** because a blended score is a third, undocumented scale nobody chose, and
  SPEC's own "measure before deciding" convention has no measurement backing a blend weight — pure
  replacement has no free parameter to get wrong.
- **A fourth step over rung 3, not a rung-4-specific merge or candidate pool,** because reranking is
  an ordering operation over a pool `retrieve_rung3` already assembles correctly; duplicating
  `merge_candidates`' provenance-union logic or growing `RetrievalResult`'s shape for a purely
  cosmetic "reranked" pool would be two places to keep a decision that lives in exactly one place.
- **Parameters over new `Settings` fields** because this ticket's own established precedent
  (`fiche_weights`, `expansion_depth`) already answers "how does an ablatable runtime knob become
  config here," and diverging from it for this one ticket, ahead of the app layer that would actually
  consume an environment variable, is the kind of forward-looking plumbing CLAUDE.md rules out.
- **FlagEmbedding for fp32, a bespoke ONNX wrapper for int8** because FlagEmbedding is already a
  direct dependency with both SPEC-named models pre-registered, while it has no ONNX backend at all —
  a second, purpose-built loader is unavoidable for that lever regardless of which library owns fp32.
- **Local quantisation over a hub-hosted quantised file** because SPEC never names one, and depending
  on an assumed filename existing under some `onnx/` prefix on the hub would make the lever fail
  silently the moment a maintainer renames or removes that export.

## Consequences

- `rerank.py` never imports `langfuse`/`langchain`, matching every other module under
  `rag.retrieval` except `langchain_retriever.py` itself — the Langfuse span still to be added there
  wraps a call to this module's `rerank()`, it does not require `rerank()` itself to change.
- `_load_reranker` caches per `(model_id, backend)` pair rather than per process the way
  `embedder.py`'s single-model `_model()` does — deliberately, since an eval run comparing rung 4's
  fp32 arm against its int8 arm needs both loaded within one process.
- The `rerank-onnx` group is untested beyond `_load_onnx_int8`'s dispatch — its real
  export/quantise/tokenize body needs `optimum`/`onnxruntime`, not installed by default, so it is
  exercised by hand before a deploy actually reaches for `RerankerBackend.ONNX_INT8`, the same
  posture `fetch_articles.py`'s real network calls already have.
- `RetrievalResult.candidate_pools` staying rung-3-shaped at rung 4 means a reader diffing rung 3 vs
  rung 4's fat objects sees identical pools and only `contexts` differ — which is the intended
  signal: rung 4 is a claim about final ordering, SPEC §12.7's fiche-primary recall@4 question, not
  about candidate depth.
