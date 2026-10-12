"""Prod tracing for the app — one Langfuse trace per question (SPEC §11.1, §14.7, #54).

On only when `Settings.langfuse_tracing` is, which `compose.yml` sets in prod and dev leaves
off (SPEC §11.2: the free tier fails on interactive debugging, not on experiments). Off, the
app builds none of this and `run_chain` runs untouched.

One trace, `ask`, holds the turn: the visitor's question and history in, the envelope out.
Inside it, the three things SPEC §11.1 asks to tell apart:

- the condenser and generation calls, as **generation** observations — LangChain's own
  callback handler, passed to both chains (`callbacks`);
- retrieval, as a first-class **retriever** observation — `retrieve` is not a LangChain
  component, so `traced_retrieve` wraps it by hand, through `run_chain`'s `retrieve_fn`
  seam, so the retrieval that runs is still the one the eval measures.

The `Langfuse` client is built from `Settings`, never from `get_client()`'s ambient
environment — the same posture `rag.eval.run_experiment` takes — and flushes on process exit
through its own `atexit` hook, which a container stopped by Sablier reaches: uvicorn exits
cleanly on SIGTERM.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient

from rag.condensation.prompt import HistoryTurn
from rag.config import Settings
from rag.generation.pipeline import GenerationResult
from rag.ingest.upsert import EmbedFn
from rag.pipeline import RetrieveFn
from rag.retrieval.candidates import Candidate
from rag.retrieval.pipeline import RetrievalResult, retrieve

__all__ = ["TRACE_NAME", "TRACE_TAG", "Tracing", "make_tracing", "untraced_turn"]

TRACE_NAME = "ask"
# What tells an app trace from an eval one in the Langfuse UI: both run `run_chain`.
TRACE_TAG = "app"


@dataclass(frozen=True)
class Tracing:
    """What a traced `make_answer_fn` threads through: `callbacks` for the two LLM chains,
    `retrieve_fn` for `run_chain`, and `turn` around each question."""

    langfuse: Langfuse
    callbacks: Sequence[BaseCallbackHandler]
    retrieve_fn: RetrieveFn

    @contextmanager
    def turn(
        self, question: str, history: Sequence[HistoryTurn]
    ) -> Iterator[Callable[[GenerationResult], None]]:
        """The `ask` trace around one question. Yields the function that records the
        result; a turn that raises ends without one, and the span records the exception."""
        with (
            self.langfuse.start_as_current_observation(
                name=TRACE_NAME,
                as_type="chain",
                input={
                    "question": question,
                    "history": [{"role": t.role, "content": t.content} for t in history],
                },
            ) as span,
            propagate_attributes(trace_name=TRACE_NAME, tags=[TRACE_TAG]),
        ):

            def record(result: GenerationResult) -> None:
                span.update(
                    output=result.envelope.model_dump(mode="json"),
                    metadata={
                        "fabricated_ids": sorted(result.citation_outcome.fabricated_ids),
                        "contexts": [_context(c) for c in result.contexts],
                    },
                )

            yield record


@contextmanager
def untraced_turn(
    question: str, history: Sequence[HistoryTurn]
) -> Iterator[Callable[[GenerationResult], None]]:
    """`Tracing.turn`'s stand-in when tracing is off: no trace, a recorder that drops it."""
    yield lambda result: None


def make_tracing(settings: Settings, *, langfuse: Langfuse | None = None) -> Tracing | None:
    """`None` when tracing is off. `langfuse` is the test seam; the real client is built
    from `settings`."""
    if not settings.langfuse_tracing:
        return None
    if langfuse is None:
        langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
            tracing_enabled=True,
        )
    return Tracing(
        langfuse=langfuse,
        # Resolves the client registered under this public key — the one built above —
        # and nests each call under whichever observation is current: the `ask` trace.
        callbacks=[CallbackHandler(public_key=settings.langfuse_public_key)],
        retrieve_fn=traced_retrieve(langfuse),
    )


def traced_retrieve(langfuse: Langfuse) -> RetrieveFn:
    """`retrieve`, inside a `retriever` observation: the query it was handed in (the
    condensed one, on a follow-up), the contexts generation will see out."""

    def retrieve_fn(
        client: QdrantClient,
        embed: EmbedFn,
        raw_turn: str,
        lookup_keys: AbstractSet[str],
        *,
        arm: str,
    ) -> RetrievalResult:
        with langfuse.start_as_current_observation(
            name="retrieval", as_type="retriever", input=raw_turn, metadata={"arm": arm}
        ) as span:
            result = retrieve(client, embed, raw_turn, lookup_keys, arm=arm)
            span.update(
                output=[_context(c) for c in result.contexts],
                metadata={"arm": arm, "short_circuit_path": result.short_circuit_path.value},
            )
            return result

    return retrieve_fn


def _context(candidate: Candidate) -> dict[str, Any]:
    """A context as the trace shows it: enough to tell which passage it was, not the text —
    the corpus is in git, and the trace has no need to carry it twice."""
    return {
        "id": candidate.id,
        "register": candidate.register.value,
        "citation_id": candidate.payload.get("citation_id") or candidate.payload.get("fiche_id"),
        "score": candidate.score,
    }
