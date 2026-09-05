"""Question → candidates: condenser · short-circuit · legs · expansion · fusion · rerank · quota (SPEC §8–§9).

`short_circuit.py` (#27) decides whether the condenser (§8, still to come) runs at all.
`candidates.py`, `legs.py`, `lookup.py` and `pipeline.py` (#28) are the rung-1 arm: plain
`qdrant-client` reads and dataclasses, no LangChain. The `BaseRetriever` subclass that
wraps `pipeline.retrieve` into a first-class LangChain/Langfuse component — converting
`Candidate` into `Document.metadata` and adding the retriever observation span — is
deliberately not built here (see the #28 issue comment: LangChain/Langfuse portions are
paired, not agent-authored). Expansion, per-leg weighting, rerank and the quota land with
their own tickets (#29-#33-ish) on top of this seam.
"""
