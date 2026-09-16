"""`(history, raw_turn)` -> a standalone query for the retriever (SPEC §8, ADR-0008, #43).

`schema.py` — the one-field output contract (`CondenserOutput`).
`prompt.py` — history formatting, the system prompt and its three few-shot examples
(`HistoryTurn`, `build_messages`).
`sanitizer.py` — the deterministic post-schema sanitizer, including reference monotonicity
(`refs(condensed) ⊆ refs(raw_turn)`, SPEC §8.4).
`pipeline.py` — the orchestration (`condense`): the `history == []` and short-circuit skips,
the sanitizer, the code-computed `condense_status`.
`chain.py` — the real `CondenseFn`: `ChatOpenAI` over OpenRouter, pinned routing, on
`CONDENSER_MODEL` — a config key independent of `GENERATION_MODEL` (SPEC §8.2).

Sits in front of the `BaseRetriever` (`rag.retrieval.langchain_retriever.PipelineRetriever`),
never inside it: the retriever needs a query *string*, and this package is what decides
which string that is. `rag.retrieval` stays exactly what its own docstring already claims —
plain `qdrant-client` reads and dataclasses, no LangChain — because the one LangChain call
in this whole path lives here instead. The condensed query never reaches generation
(SPEC §8.5, ADR-0009's history stripping): `rag.generation` never imports this package.
"""
