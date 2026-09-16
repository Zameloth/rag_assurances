"""The real `CondenseFn`: `ChatOpenAI` pointed at OpenRouter, pinned routing,
`with_structured_output()` over `CondenserOutput` (SPEC §8.2, §8.3, ADR-0008, #43).

Mirrors `rag.generation.chain`'s own shape one seam over: `CondenseFn` is declared in
`pipeline.py`, and this module is the sibling that actually imports `langchain_openai` —
`pipeline.py` stays free of any concrete LLM SDK, same split `rag.ingest.upsert`/
`embedder.py` and `rag.generation.pipeline`/`chain.py` already use.

Pinned to `settings.condenser_model`/`settings.condenser_provider` — a config key
independent of `GENERATION_MODEL` (SPEC §8.2, ADR-0008): the generation model is an
ablatable arm, and the golden set's 10 multi-turn items are the only items measuring the
condenser at all, so a shared key would silently swap the condenser on exactly those items
whenever the generation arm changes. `CondenserOutput` is already a plain object at its
schema root (`schema.py`), unlike generation's `Reponse | Refus` union, so no
`_EnvelopeContainer`-style wrapper is needed here — `with_structured_output` takes it
directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag.condensation.pipeline import CondenseFn
from rag.condensation.prompt import Message
from rag.condensation.schema import CondenserOutput
from rag.config import Settings

__all__ = ["OPENROUTER_BASE_URL", "make_condense_fn"]

# Not a `Settings` field, same reasoning `rag.generation.chain.OPENROUTER_BASE_URL` already
# states: OpenRouter's endpoint doesn't move with the environment the way the Qdrant/
# Langfuse URLs do.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_ROLE_TO_MESSAGE_CLS: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def _to_langchain_messages(messages: list[Message]) -> list[BaseMessage]:
    """`build_messages`'s plain `(role, content)` pairs -> LangChain message objects, same
    order — identical mapping to `rag.generation.chain._to_langchain_messages`, kept as its
    own copy rather than imported: this package must not depend on `rag.generation` (see
    the package docstring), even for a two-line helper the two happen to share verbatim."""
    return [_ROLE_TO_MESSAGE_CLS[role](content) for (role, content) in messages]


def _build_chain(settings: Settings) -> Runnable[list[BaseMessage], CondenserOutput]:
    """The `ChatOpenAI` client pinned to `settings.condenser_model` /
    `settings.condenser_provider`, wrapped in `with_structured_output(CondenserOutput,
    strict=True)` (SPEC §8.3: `response_format: {type: json_schema, strict: true}` under
    pinned routing — `provider.require_parameters`, `allow_fallbacks: false`, a pinned
    `order`, the same shape `rag.generation.chain._build_chain` already pins for
    `GENERATION_MODEL`).
    """
    if not settings.condenser_model:
        raise RuntimeError("condenser_model not provided")
    if not settings.condenser_provider:
        raise RuntimeError("condenser_provider not provided")
    if not settings.openrouter_api_key:
        raise RuntimeError("openrouter_api_key not provided")

    # Same `cast` reasoning as `rag.generation.chain._build_chain`: `with_structured_output`'s
    # own signature (`schema: dict | type[_BM] | type`) makes mypy settle for `dict | Any`
    # on the bare `type` alternative instead of binding `_BM` to `CondenserOutput`, even
    # though the docstring guarantees a `CondenserOutput` instance back for a Pydantic
    # schema class.
    return cast(
        Runnable[list[BaseMessage], CondenserOutput],
        ChatOpenAI(
            api_key=SecretStr(settings.openrouter_api_key),
            model=settings.condenser_model,
            base_url=OPENROUTER_BASE_URL,
            extra_body={
                "provider": {
                    "require_parameters": True,
                    "allow_fallbacks": False,
                    "order": [settings.condenser_provider],
                }
            },
        ).with_structured_output(CondenserOutput, strict=True),
    )


@lru_cache(maxsize=1)
def _cached_chain(settings: Settings) -> Runnable[list[BaseMessage], CondenserOutput]:
    """One chain per distinct `Settings` — `Settings` is a frozen dataclass, so it hashes,
    and rebuilding a `ChatOpenAI` client per call would be pure waste (mirrors
    `rag.generation.chain._cached_chain`)."""
    return _build_chain(settings)


def make_condense_fn(settings: Settings) -> CondenseFn:
    """The `CondenseFn` `rag.condensation.pipeline.condense` expects: `list[Message] ->
    CondenserOutput`. Built once per `settings`, reused across calls.

    Any call failure (timeout, pinned endpoint down, a malformed response) propagates as
    whatever exception the chain raises — `condense()` is what catches it and turns it into
    `CondenseStatus.FALLBACK_ERROR` (SPEC §8.3); this function does not swallow anything
    itself, so a test can tell the two apart.
    """
    chain = _cached_chain(settings)

    def condense_fn(messages: list[Message]) -> CondenserOutput:
        lc_messages = _to_langchain_messages(messages)
        return chain.invoke(lc_messages)

    return condense_fn
