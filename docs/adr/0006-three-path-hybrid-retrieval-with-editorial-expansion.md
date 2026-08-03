# ADR-0006 — Three retrieval paths, hybrid on both legs, with the consumer→legal hop removed structurally

- **Status**: Accepted — 2026-08-03
- **Tickets**: [#9](https://github.com/Zameloth/rag_assurances/issues/9), corrected by [#12](https://github.com/Zameloth/rag_assurances/issues/12) and [#15](https://github.com/Zameloth/rag_assurances/issues/15)
- **Spec**: [`SPEC.md` §9](../../SPEC.md#9-retrieval)

## Context

The answer contract needs both a fiche and an article, but the query arrives in consumer French
and the best open embedder manages **~25 nDCG@10 on legal French**. Asking one index to bridge
that gap is asking it to do the thing it is measurably worst at.

## Decision

**Three paths, hybrid on both search legs, cross-encoder rerank, register-quota assembly.**

1. **Fiche leg** — hybrid over `fiches`, top-20, **dense-leaning**.
2. **`<dc:source>` expansion** — top-3 fiches → their sections → a **filtered vector search**
   within them, capped at 40.
3. **Article leg** — hybrid over `articles`, top-20, **sparse-leaning**.

Plus an **article-reference short-circuit** that skips search entirely, a hand-wrapped
`bge-reranker-v2-m3` stage (**ablatable**), and a **4 fiche + 4 article quota** with a relevance
floor and an explicit no-article marker.

## Rationale

- **Path 2 is the spine and the thesis.** It does not improve the weak hop, it **removes** it —
  DILA has already asserted editorially which sections each fiche rests on, so following that
  link is a lookup, not a guess. Rung 3 is the headline experiment, placed early enough that a
  negative result can still change the design.
- **Path 3 exists because 2,377 of ~2,464 documents are articles** — most of the corpus is
  unreachable if the only way in is via one of 87 fiches.
- **The two legs get different weights** because the M3 paper's French numbers **invert with
  document length**: MIRACL-fr dense 78.6 / sparse 65.4, MLDR-fr dense 73.8 / sparse 82.7.
  Fusion is **client-side**, because RRF merges by rank without weights.
- **M3-sparse, not BM25.** BM25 is easier to debug and handles article numbers, but the 82.7 is
  *M3-sparse specifically*, and the article-number weakness is answered by the short-circuit
  regardless.
- **The short-circuit came from measuring the tokenizer**: `L113-2` and `L. 113-2` tokenize
  differently and **do not match each other**; `A.121-1` is mangled outright. No sparse weighting
  fixes this, and nothing needs to — the number is a metadata field.
- **The quota designs out an intermittent failure**: naive top-8 can return eight fiche chunks
  and zero articles, leaving nothing to cite on the system's headline feature.

## Consequences

- **Two corrections landed on this design, both silent-wrong-answer class.** The original
  short-circuit regex was **broken on 21% of the corpus** — `L113-15-2` truncates to `L113-15`,
  *itself a real in-force article* — and the cap of 40 had **no selection rule**, so the default
  `scroll` would have ordered by UUID hash and made rung 3 partly measure the hash function. Both
  are fixed in [ADR-0007](0007-flat-payload-two-indexes-cid-identity.md); the query-side scanner
  needs a *different* pattern from the field validator.
- **Rung 5 is scored on zero-article rate**, not mean article recall — the quota was never built
  to raise the mean.
- **The reranker is not a LangChain component** and must be hand-wrapped in a span, or rung 4
  measures nothing. It is also a **RAM lever**, not only a quality lever.
- Sparse support becomes mandatory on the store ([ADR-0005](0005-qdrant-two-collections-behind-aliases.md)),
  and fiche chunks must carry their section ids or expansion has nothing to follow.
- The floor-plus-marker converts "bad article" into "no article, stated plainly" — a stated
  outcome rather than a silent one.
