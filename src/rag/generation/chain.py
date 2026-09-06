"""The real `GenerateFn`: `ChatOpenAI` pointed at OpenRouter, pinned routing,
`with_structured_output()` over `Envelope` (SPEC §10.1, §10.2, ADR-0009, #42).

Mirrors `rag.ingest`'s own split: `EmbedFn` is declared in `upsert.py` but the real model
call lives in a sibling module, `embedder.py`, which `upsert.py` never imports from. Same
shape here — `GenerateFn` is declared in `pipeline.py`, and this module is the sibling that
actually imports `langchain_openai`; `pipeline.py` stays free of any concrete LLM SDK.

This is the LangChain portion of #42 — paired with @Zameloth rather than agent-authored
(the issue's collaboration note, the same line #31's Langfuse span drew). What's below is
the skeleton only: signatures and the seam, not the chain itself.
"""

from __future__ import annotations

from functools import lru_cache
from typing import cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from rag.config import Settings
from rag.generation.pipeline import GenerateFn
from rag.generation.prompt import Message
from rag.generation.schema import Envelope

__all__ = ["OPENROUTER_BASE_URL", "make_generate_fn"]

# Not a `Settings` field, unlike the Qdrant/Langfuse URLs — those genuinely differ dev vs
# prod; OpenRouter's endpoint doesn't move with the environment.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_ROLE_TO_MESSAGE_CLS: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


class _EnvelopeContainer(BaseModel):
    """Racine `object` valide pour `strict: true` (OpenAI interdit un `anyOf` en racine) —
    l'union discriminée `Envelope` vit comme propriété interne, jamais exposée hors de ce
    module."""

    envelope: Envelope


def _to_langchain_messages(messages: list[Message]) -> list[BaseMessage]:
    """`build_messages`'s plain `(role, content)` pairs -> LangChain message objects, same
    order. TODO(@Zameloth): straightforward mapping via `_ROLE_TO_MESSAGE_CLS`, but confirm
    against this SDK version that `AIMessage` is really what an 'assistant' history turn
    should become here."""

    return [_ROLE_TO_MESSAGE_CLS[role](content) for (role, content) in messages]


def _build_chain(settings: Settings) -> Runnable[list[BaseMessage], _EnvelopeContainer]:
    """The `ChatOpenAI` client pinned to `settings.generation_model` /
    `settings.generation_provider`, wrapped in `with_structured_output(_EnvelopeContainer,
    strict=True)` — `_EnvelopeContainer`, not `Envelope` directly, since OpenAI's `strict`
    mode rejects an `anyOf` at the schema root and `Envelope` is exactly that.

    Still open: `extra_body`'s `provider` block reaching the real outbound request has
    only been checked against `with_structured_output`'s signature, not a live call — worth
    confirming against the actual HTTP body before trusting it. `RuntimeError` below covers
    only `generation_model`; `generation_provider`/`openrouter_api_key` being empty aren't
    checked yet.
    """
    if not settings.generation_model:
        raise RuntimeError("generation_model not provided")
    if not settings.generation_provider:
        raise RuntimeError("generation_provider not provided")
    if not settings.openrouter_api_key:
        raise RuntimeError("openrouter_api_key not provided")

    # `with_structured_output`'s own signature takes `schema: dict | type[_BM] | type`
    # (langchain_openai's chat_models/base.py) — the bare `type` alternative makes mypy
    # settle for `dict | Any` instead of binding `_BM` to `_EnvelopeContainer`, even though
    # the docstring guarantees a Pydantic instance back when `schema` is a Pydantic class.
    # `cast` states that runtime contract instead of letting the `Any` propagate silently.
    return cast(
        Runnable[list[BaseMessage], _EnvelopeContainer],
        ChatOpenAI(
            api_key=SecretStr(settings.openrouter_api_key),
            model=settings.generation_model,
            base_url=OPENROUTER_BASE_URL,
            extra_body={
                "provider": {
                    "require_parameters": True,
                    "allow_fallbacks": False,
                    "order": [settings.generation_provider],
                }
            },
        ).with_structured_output(_EnvelopeContainer, strict=True),
    )


@lru_cache(maxsize=1)
def _cached_chain(settings: Settings) -> Runnable[list[BaseMessage], _EnvelopeContainer]:
    """One chain per distinct `Settings` — `Settings` is a frozen dataclass, so it hashes,
    and rebuilding a `ChatOpenAI` client per call would be pure waste."""
    return _build_chain(settings)


def make_generate_fn(settings: Settings) -> GenerateFn:
    """The `GenerateFn` `rag.generation.pipeline.generate` expects: `list[Message] ->
    Envelope`. Built once per `settings`, reused across calls."""
    chain = _cached_chain(settings)

    def generate_fn(messages: list[Message]) -> Envelope:
        # `.envelope` unwraps `_EnvelopeContainer` — the container never leaves this module.
        lc_messages = _to_langchain_messages(messages)
        return chain.invoke(lc_messages).envelope

    return generate_fn
