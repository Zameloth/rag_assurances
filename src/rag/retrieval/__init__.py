"""Question → candidates: condenser · short-circuit · legs · expansion · fusion · rerank · quota (SPEC §8–§9).

`short_circuit.py` (#27) decides whether the condenser (§8, still to come) runs at all.
`candidates.py`, `legs.py`, `lookup.py` and `pipeline.py` (#28) are the rung-1 arm; `fusion.py`
(#29) adds rung 2's per-leg-weighted dense+sparse fusion on top of the same seam, and
`expansion.py` (#30) adds rung 3's `<dc:source>` expansion — the headline experiment — as a
third, additive candidate pool on top of that. `rerank.py` (#31) adds rung 4's cross-encoder
over rung 3's exact fused pool. All of it is plain `qdrant-client`/FlagEmbedding reads and
dataclasses, no LangChain. The `BaseRetriever` subclass that wraps `pipeline.retrieve` into a
first-class LangChain/Langfuse component — converting `Candidate` into `Document.metadata` and
adding the retriever observation span — is deliberately not built out fully here (see the #28
issue comment: LangChain/Langfuse portions are paired, not agent-authored); the reranker's own
hand-wrapped Langfuse span (SPEC §11.1 — it doesn't auto-trace, not being a LangChain component)
is the same kind of paired work, still to land in `langchain_retriever.py`. The quota lands with
its own ticket (#33-ish) on top of this seam.
"""
