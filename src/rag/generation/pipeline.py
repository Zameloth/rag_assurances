"""`RetrievalResult` -> the fat generation object: prompt, model call, citation guardrail
(SPEC §10, §10.7, #42).

`generate_fn` is the seam a real `with_structured_output()` LangChain chain plugs into —
the same shape `RerankFn`/`EmbedFn` already take (`rag.retrieval.rerank`,
`rag.ingest.upsert`). Building that real chain is the LangChain portion of this ticket:
OpenRouter's pinned routing (SPEC §10.1 — `provider.require_parameters`,
`allow_fallbacks: false`, a pinned `order`) and `response_format: {type: json_schema,
strict: true}` wrapped in `with_structured_output()`. That part is paired with @Zameloth
rather than agent-authored (the #42 issue comment draws the same line #31's Langfuse span
did), so it is not built here — this module owns everything around that seam: the prompt,
the guardrail, and the shape callers get back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rag.generation.citation import CitationOutcome, check_citations
from rag.generation.prompt import HistoryTurn, Message, build_messages
from rag.generation.schema import Envelope
from rag.retrieval.candidates import Candidate
from rag.retrieval.pipeline import RetrievalResult

__all__ = ["GenerateFn", "GenerationResult", "generate"]

# `Message` (a plain `(role, content)` pair, `rag.generation.prompt`) in, an `Envelope` out
# — the real implementation runs `build_messages`'s output through a `with_structured_output`
# chain pinned to `GENERATION_MODEL` (SPEC §10.1). Tests inject a fake, mirroring `RerankFn`.
GenerateFn = Callable[[list[Message]], Envelope]


@dataclass(frozen=True)
class GenerationResult:
    """The fat object (SPEC §10.7): the typed envelope, the citation-check outcome, the
    final contexts and the per-leg candidate pools — everything an evaluator needs, since
    Langfuse evaluators see only this return value, never the trace."""

    envelope: Envelope
    citation_outcome: CitationOutcome
    contexts: list[Candidate]
    candidate_pools: dict[str, list[Candidate]] = field(default_factory=dict)


def generate(
    raw_turn: str,
    retrieval: RetrievalResult,
    generate_fn: GenerateFn,
    *,
    history: Sequence[HistoryTurn] = (),
) -> GenerationResult:
    """Build the prompt from `retrieval.contexts`, call `generate_fn`, check citations
    against the same contexts.

    Never repairs (SPEC §10.5): the demo path
    (`rag.generation.citation.repair_for_display`) is a caller's explicit choice on the
    returned envelope, not something this function does on the way out — eval always sees
    the model's own, unrepaired output.
    """
    messages = build_messages(raw_turn, retrieval.contexts, history)
    envelope = generate_fn(messages)
    outcome = check_citations(envelope, retrieval.contexts)
    return GenerationResult(
        envelope=envelope,
        citation_outcome=outcome,
        contexts=retrieval.contexts,
        candidate_pools=retrieval.candidate_pools,
    )
