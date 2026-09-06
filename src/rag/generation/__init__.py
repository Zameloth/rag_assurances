"""Candidates → answer: schema · prompt · citation guardrail (SPEC §10, #42).

`schema.py` — the typed answer envelope (`Envelope`, `Reponse`, `Refus`, `Motif`).
`prompt.py` — context-by-register assembly and history stripping (`build_messages`).
`citation.py` — the `cited ⊆ retrieved_context` guardrail (`check_citations`).
`pipeline.py` — the fat `GenerationResult` object (`generate`).
`chain.py` — the real `GenerateFn`: `ChatOpenAI` over OpenRouter, pinned routing (SPEC
§10.1), `with_structured_output()` over `Envelope` (`make_generate_fn`). Paired with
@Zameloth rather than agent-authored (issue #42 comment), the same division #31's
Langfuse span drew — currently a skeleton, not the finished chain.
"""
