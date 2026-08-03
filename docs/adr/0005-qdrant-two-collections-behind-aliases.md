# ADR-0005 — Qdrant on Docker, two collections behind aliases, UUIDv5 point ids on `cid`

- **Status**: Accepted — 2026-08-03
- **Tickets**: [#8](https://github.com/Zameloth/rag_assurances/issues/8), corrected by [#15](https://github.com/Zameloth/rag_assurances/issues/15)
- **Spec**: [`SPEC.md` §6](../../SPEC.md#6-vector-store)

## Context

Corpus volume (~5 MB) is small enough that every candidate store handles it comfortably, so
volume should not drive the decision. The binding constraint is that retrieval needs **BGE-M3's
own learned sparse weights**, supplied as raw indices and values — not sparse vectors the store
generates for itself.

## Decision

**Qdrant, as a Docker service in dev and on the VPS alike, driven through the native client
behind one custom `BaseRetriever`.** Two collections (`fiches`, `articles`) behind **stable
aliases**, dense + sparse as **named vectors on one point**, **UUIDv5** point ids over the
natural key, and `indexing_threshold=0` everywhere.

```
articles: uuid5(NS, f"{legiarti_cid}#{chunk_index}")
fiches:   uuid5(NS, f"{fiche_id}#{chunk_index}")
```

## Rationale

- **The store is a query engine, not a `VectorStore`.** Every high-level hybrid helper on offer
  **fuses for you**, which is exactly what per-leg weighting forbids — accepting server-side RRF
  would silently discard the evidence the design turns on and make rung 2 untestable. So
  "LangChain integration maturity" drops out as a criterion; the wrapper keeps Langfuse tracing
  anyway.
- **The field**: Chroma shipped sparse in Nov 2025 but generates its own rather than accepting
  ours; LanceDB has BM25 FTS only; pgvector caps HNSW at 1,000 non-zeros, which a long article
  clears; Milvus is viable but needs 3 compose services and recommends 16 GB, on a box already
  holding two 568M models.
- **`indexing_threshold=0` is measured, not stylistic.** The default builds HNSW past ~10 MB of
  vectors, i.e. ~2,500 points at 1024-dim fp32 — and the two collections **straddle it** (11.5 MB
  and 3.6 MB). Without the override, the two legs of one query would run **different search
  algorithms as a side effect of collection size**, inside a ladder built to attribute each delta
  to one variable. Embedded mode would have reintroduced the same confound, since local mode is a
  pure-Python reimplementation, not the Rust engine.
- **Two collections make the topology literal**: separate weights, pools, metrics and quota slots,
  with no possibility of a fiche leaking into an article lookup.
- **Aliases make an arm swap atomic**, so no experiment or application code ever names a physical
  collection.
- **UUIDv5 buys idempotent ingestion and store-free id derivation**, which is what keeps gold
  labels valid across every arm.

## Consequences

- **`cid`, not the version id.** 52% of articles have been amended, and under the version id an
  amended article hashes to a *different point*: the upsert writes a new point and leaves the
  stale one behind as a **retrievable orphan** — the exact failure UUIDv5 was chosen to prevent,
  and silent. `legiarti_version_id` stays as payload, and is the correct id for the Légifrance
  URL.
- `QdrantClient(":memory:")` is kept **for pytest only**.
- Re-enabling HNSW later becomes its own clean experiment.
- Aliases pay a second time under container sleep: a restore killed mid-upsert never acquires the
  alias ([ADR-0014](0014-parquet-points-dump-on-a-release-with-alias-flip.md)).
- `indexing_threshold=0` has a consequence nobody costed at the time: it removes the only reason
  to prefer a Qdrant snapshot as a delivery format.
