# ADR-0022 — Rung 6's e5 arm is a combined `EmbedFn`, explicit mean pooling, and RAM measured query-shaped

- **Status**: Accepted
- **Ticket**: #39
- **Spec**: SPEC.md §4, §5, §6.4, §14.4, §15.7, ADR-0004

## Context

SPEC §5 and ADR-0004 already settle rung 6's shape — `intfloat/multilingual-e5-large-instruct`
dense vectors, BGE-M3 sparse vectors on the same points, no re-chunking, shared point ids with the
incumbent — and #39's acceptance criteria add three things neither document pins down:

1. **How e5's dense half and M3's sparse half become one `EmbedFn`.** `upsert.py`'s `EmbedFn`
   shape (`Sequence[str] -> list[Embedding]`) is the one interface every ingest and query caller
   already depends on (ADR-0021) — rung 6 needs a function of that exact shape that internally
   runs two different models, not a new interface.
2. **e5-instruct's own asymmetry.** Unlike BGE-M3's single symmetric call, e5-instruct puts an
   instruction prefix on queries only. Getting the query/passage sides backwards, or picking the
   wrong pooling method, is not the kind of bug that raises — it silently degrades recall.
3. **What "resident memory with both embedders loaded" means operationally** — a number has to
   come from somewhere concrete, at some concrete point in a process's life.

## Decision

**Two arm collections, `articles__e5-m3__c512__v1` / `fiches__e5-m3__c512__v1`**, built by
`rag.ingest.e5_arm.build_articles_e5_arm`/`build_fiches_e5_arm` — the same shape
`rag.ingest.ab_arms`'s challenger builders already take: read the same committed
`articles_path`/`fiches_dir`, `ensure_*_collection`, `upsert_*` with `dense_text=None` (no
enrichment seam — rung 6 changes the embedder, not the embedded text, so `upsert.py`'s two-call
`dense_text` mechanism from #38 doesn't apply here). Neither builder flips
`ARTICLES_ALIAS`/`FICHES_ALIAS` — that is #41's job, once a verdict is read.

**`rag.ingest.e5_embedder` owns the e5+M3 combination**, as four functions, never one function
with a bool flag:

- `embed_passages`/`embed_queries` — e5-only dense vectors, via `FlagModel.encode_corpus`/
  `encode_queries` respectively. Only `encode_queries` applies an instruction prefix
  (`query_instruction_format="Instruct: {}\nQuery: {}"`, `query_instruction_for_retrieval` set to
  the model card's own example retrieval task text) — matching `FlagEmbedding`'s own `E5_MAPPING`
  entry for this model id and the model's documented usage ("no need to add instruction for
  retrieval documents").
- `index_embed_batch`/`query_embed_batch` — the real `EmbedFn`s: e5's passage/query dense half
  stitched onto BGE-M3's sparse half (`rag.ingest.embedder.embed_batch`, dense half discarded),
  used respectively by the arm builders and by whatever calls `retrieve_rungN`/
  `run_retrieval_ladder` for rung 6.

**`FlagModel` is constructed with `pooling_method="mean"`, explicitly.** `FlagModel`'s own
default is CLS pooling (`AbsEmbedder.DEFAULT_POOLING_METHOD`); `multilingual-e5-large-instruct` is
architecturally mean-pooled, and `FlagEmbedding`'s own `E5_MAPPING` entry for this model id pins
`mean` — but only `FlagAutoModel.from_finetuned` reads that mapping automatically.
`rag.ingest.embedder`'s own precedent constructs `BGEM3FlagModel` directly rather than through the
auto-loader (that model needs no pooling override, so the gap never surfaced there); doing the
same for e5 without the explicit override would have silently run every dense vector through the
wrong pooling — a code-review-visible one-line fix once caught, not a design question, but exactly
the kind of thing "prefix conventions handled correctly" (#39's own acceptance criterion) has to
mean here.

**Resident memory is measured by a dedicated, build-free script,
`scripts/measure_e5_ram.py`, never by `scripts/build_e5_arm.py`.** `ru_maxrss` (`rag.eval.
resident_memory.measure_resident_memory_mb`) is a high-water mark that never falls for the life of
a process (`man 2 getrusage`). SPEC §15.7 frames the number this ticket needs as "two embedding
models co-resident **at query time**" — but `build_e5_arm.py` runs `upsert.py`'s 500-point ingest
batches, whose transient batch-embedding activations would bake a one-time bulk-ingest peak into a
number meant to describe steady query-time serving. `measure_e5_ram.py` instead loads both models
by calling `query_embed_batch` once, batch size 1, on one representative question — the exact
shape every `retrieve_rungN` call already embeds (`embed([raw_turn])[0]`) — then reads `ru_maxrss`
from that same, otherwise-idle process. The result is written to
`eval/runs/rung6-resident-memory.json` (`rag.eval.resident_memory.write_resident_memory_report`,
the same machine-written-JSON shape `rag.eval.retrieval_run.write_run` already established) so
#41's ladder run can cite it next to rung 6's quality verdict instead of re-measuring.

## Rationale

- **Four named functions over one parameterised one** (`embed_passages`/`embed_queries`,
  `index_embed_batch`/`query_embed_batch`) because e5-instruct's query/passage asymmetry is a
  design constraint stated in the model's own card, not an implementation detail — collapsing it
  into a flag would make "did this call use the wrong side by mistake" invisible in a diff and
  silent in behaviour (a recall regression, not an exception).
- **No `dense_text` seam reused from #38** because that seam exists to make *one* embedder answer
  a batch twice, on two different texts; rung 6 makes *two* embedders answer a batch once each, on
  the same text. Reusing the seam's signature for a problem it wasn't shaped for would be the wrong
  abstraction, not a smaller diff.
- **Explicit `pooling_method="mean"` over switching to `FlagAutoModel.from_finetuned`** because
  `rag.ingest.embedder` already established the direct-construction pattern for BGE-M3, and this
  module's whole job is combining two direct constructions into one `EmbedFn` shape — introducing
  a second loading mechanism for only one of the two models would be a second thing to reconcile
  for no benefit `from_finetuned` provides here beyond the one config lookup it would save.
- **A separate script for the RAM measurement, not a flag on the build script**, because the two
  numbers a combined script would produce (bulk-ingest peak vs. query-time-shaped peak) answer
  different questions, and only one of them is SPEC §15.7's — a single script silently reporting
  the wrong one would look identical to reporting the right one until someone read the number
  against a deploy-time OOM.

## Measured result

`scripts/build_e5_arm.py` was run against the full committed corpus (2,375 article rows, 87
fiches) on 2026-09-16, producing `articles__e5-m3__c512__v1` (2,801 points) and
`fiches__e5-m3__c512__v1` (849 points) — the same point counts *and the same point ids* as the
incumbent `articles`/`fiches` arms (verified by scrolling both collection pairs and diffing id
sets), confirming the shared-chunk-population requirement holds in practice, not just by
construction. `scripts/measure_e5_ram.py` was then run fresh (no prior model loaded in that
process) and recorded:

**4,574 MB (≈4.47 GiB) peak RSS** with BGE-M3 and `multilingual-e5-large-instruct` both loaded
and warmed by one query-sized embed call each. Against SPEC §14.4's ~4.5 GB prod budget (with the
other demo awake), the two embedders alone already consume essentially all of it — **≈34 MiB of
headroom**, before Qdrant, the reranker every real query also needs resident (rung 4's own arm,
568M–306M params depending on the RAM lever chosen — SPEC §14.4), or the app/HTTP runtime's own
overhead on top of what `ru_maxrss` already counts here. In practice that reads as "does not fit
alongside a serving process's other residents," even though the two-embedder number alone is
technically still under 4.5 GB. This is read alongside rung 6's quality verdict in #41, not decided
here: SPEC §15.7's pre-registered rule is "prod runs the ladder-winning arm if it fits; else the
cheapest arm that does," so a RAM-budget miss does not by itself veto rung 6 winning on quality —
it determines what *ships*, a decision #41 makes once the quality verdict is in.

Full report: `eval/runs/rung6-resident-memory.json`.

## Consequences

- Both e5 arm collections are left built in the dev Qdrant instance for reproducibility/
  inspection, the same posture ADR-0021 left the two pre-ladder challenger collections in; neither
  alias points at them, and no code outside `rag.ingest.e5_arm`/`scripts/build_e5_arm.py` names
  them.
- `rag.ingest.e5_embedder.query_embed_batch` is now the seam #41's rung-6 eval run passes as
  `embed` to `run_retrieval_ladder`/`retrieve_rungN` — no change needed in
  `rag.retrieval.pipeline` itself, since every rung already takes `embed: EmbedFn` as an injected
  parameter rather than importing a concrete embedder.
- The RAM figure recorded here is a single-process, single-query measurement, not a load-test
  peak under concurrent requests — consistent with CONTEXT.md's own framing of the RAM budget as
  "the constraint is concurrency, not capacity": the number #41/§15.7's rule needs is exactly "one
  request's worth of both models resident," since the OOM risk *is* concurrent requests each
  paying this same fixed cost, not one request paying more than this.
