"""Question → candidates: condenser · short-circuit · legs · expansion · fusion · rerank · quota (SPEC §8–§9).

`short_circuit.py` (#27) decides whether the condenser (§8, still to come) runs at all.
`candidates.py`, `legs.py`, `lookup.py` and `pipeline.py` (#28) are the rung-1 arm; `fusion.py`
(#29) adds rung 2's per-leg-weighted dense+sparse fusion on top of the same seam, and
`expansion.py` (#30) adds rung 3's `<dc:source>` expansion — the headline experiment — as a
third, additive candidate pool on top of that. All of it is plain `qdrant-client` reads and
dataclasses, no LangChain. The `BaseRetriever` subclass that wraps `pipeline.retrieve` into a
first-class LangChain/Langfuse component — converting `Candidate` into `Document.metadata` and
adding the retriever observation span — is deliberately not built here (see the #28 issue
comment: LangChain/Langfuse portions are paired, not agent-authored). Rerank and the quota land
with their own tickets (#32-#33-ish) on top of this seam.
"""
