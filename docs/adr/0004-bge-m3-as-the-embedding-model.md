# ADR-0004 — BGE-M3 as the embedding model, with multilingual-e5 as the ablated arm

- **Status**: Accepted — 2026-08-03
- **Ticket**: [#4](https://github.com/Zameloth/rag_assurances/issues/4)
- **Spec**: [`SPEC.md` §5](../../SPEC.md#5-embeddings)

## Context

Embeddings run locally, so they are free to re-run over the whole corpus and cheap to A/B. The
corpus is **legal and administrative French**, which is a different problem from general French.

MTEB(fra) v1 has exactly five retrieval tasks, and only **BSARD** (Belgian statutory articles,
jurist-written queries) tests legal French. Syntec is a trap: it looks like the ideal proxy but
its authors state its language "does not feature the specificity of the legal vocabulary", and
at ~90 documents it is near-saturated.

## Decision

**Default `BAAI/bge-m3`; runner-up `intfloat/multilingual-e5-large-instruct`, ablated at rung 6
as e5-dense + M3-sparse. No ColBERT.**

## Rationale

- **French-specific ≠ legal-capable.** `Solon-embeddings-large-0.1` — the obvious "it's the
  French one" pick — scores BSARD nDCG@10 **2.08** while scoring 84.60 on Syntec.
- BGE-M3 is MIT, 568M, 1024-dim, CPU-viable, and the **only candidate emitting dense + sparse +
  ColBERT from one forward pass**, which turns hybrid retrieval into a config change rather than
  a re-embed.
- **Stated weakness, not hidden**: it has *no* MTEB(fra) retrieval numbers at all. It is
  recommended partly on architectural fit. The most on-point evidence is its own paper's French
  column, where the dense/sparse relationship **inverts with document length** — which is what
  the two-leg weighting later turns on.
- **ColBERT is rejected on the paper's own French numbers**: MLDR-fr dense+sparse 84.2 vs
  all-three 83.9 — it *subtracts*, while costing ~1 vector per token in storage.
- Ruled out: CamemBERT-derived (`max_seq_length: 128`), `mistral-embed` (still version 23.12, no
  traceable French number), `jina-embeddings-v3` (CC-BY-NC-4.0), `bge-multilingual-gemma2` (best
  numbers measured, but 9B and not CPU-viable).

## Consequences

- **Calibrate expectations downward.** The best open local model reaches ~25 nDCG@10 on legal
  French against 82–88 on general French. **The retrieval stack has to do the work, not the
  embedder** — which is why [ADR-0006](0006-three-path-hybrid-retrieval-with-editorial-expansion.md)
  removes the consumer→legal hop structurally rather than trying to improve it.
- **e5 emits no sparse vectors**, so rung 6's arm must be e5-dense + M3-sparse; running e5 alone
  would change two variables and un-do rung 2. Both models are resident for that rung — the one
  index-bearing arm the RAM budget could veto.
- The warning above is about **embedders**, which must *find* legal text by similarity. It does
  **not** transfer to generation, which never performs that hop — which is why no research ticket
  was spent de-risking the generation model.
- **Currency risk, recorded**: Qwen3-Embedding, EmbeddingGemma and granite-r2 have never been
  benchmarked on French retrieval. A newer model may already beat both picks unmeasured.
