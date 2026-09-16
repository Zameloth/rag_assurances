"""The real `CondenseFn` seam (`rag.condensation.chain`, SPEC §8.2, §8.3, #43), against a
fake `ChatOpenAI`/`with_structured_output` chain.

Never calls OpenRouter: the contract under test is message conversion, the pinned-routing
`extra_body` shape, per-`Settings` chain caching, and that `CondenserOutput` (not some
wrapper) is what's handed to `with_structured_output` — not the model's own rewrite
quality, which only a live call (a smoke script, not the suite) can speak to.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import rag.condensation.chain as chain_module
from rag.condensation.chain import OPENROUTER_BASE_URL, _to_langchain_messages
from rag.condensation.prompt import Message
from rag.condensation.schema import CondenserOutput
from rag.config import Settings


def _settings(**overrides: Any) -> Settings:
    fields: dict[str, Any] = {
        "openrouter_api_key": "sk-or-test",
        "generation_model": "",
        "generation_provider": "",
        "condenser_model": "mistralai/mistral-small-3.2-24b-instruct",
        "condenser_provider": "mistral/eu",
        "judge_model": "",
        "judge_provider": "",
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
        "langfuse_base_url": "https://cloud.langfuse.com",
        "langfuse_tracing": False,
        "qdrant_url": "http://localhost:6333",
    }
    fields.update(overrides)
    return Settings(**fields)


class _FakeStructuredChain:
    def __init__(self, schema: type) -> None:
        self.schema = schema


class _FakeChatOpenAI:
    """Records constructor kwargs; `.with_structured_output` records the schema/`strict`
    it was given and hands back a stand-in — never touches the network."""

    instances: list[_FakeChatOpenAI] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.structured_schema: type | None = None
        self.structured_strict: bool | None = None
        _FakeChatOpenAI.instances.append(self)

    def with_structured_output(
        self, schema: type, *, strict: bool | None = None
    ) -> _FakeStructuredChain:
        self.structured_schema = schema
        self.structured_strict = strict
        return _FakeStructuredChain(schema)


@pytest.fixture(autouse=True)
def _fake_chat_openai(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Swaps in the fake class and clears both the instance log and the `_cached_chain`
    `lru_cache` singleton on both sides of the test — mirrors
    `tests/generation/test_generation_chain.py`'s own fixture, for the same leaked-cache
    risk (`test_embedder.py`'s BGE-M3 singleton guard)."""
    _FakeChatOpenAI.instances.clear()
    monkeypatch.setattr(chain_module, "ChatOpenAI", _FakeChatOpenAI)
    chain_module._cached_chain.cache_clear()
    yield
    chain_module._cached_chain.cache_clear()


# --- _to_langchain_messages --------------------------------------------------


def test_to_langchain_messages_maps_roles_in_order() -> None:
    messages: list[Message] = [
        ("system", "instructions"),
        ("user", "Et si je suis locataire ?"),
        ("assistant", "Oui, sous conditions."),
    ]

    result = _to_langchain_messages(messages)

    assert [type(m).__name__ for m in result] == ["SystemMessage", "HumanMessage", "AIMessage"]
    assert [m.content for m in result] == [
        "instructions",
        "Et si je suis locataire ?",
        "Oui, sous conditions.",
    ]


# --- _build_chain: the consumer-side checks -----------------------------------


def test_build_chain_raises_when_condenser_model_missing() -> None:
    with pytest.raises(RuntimeError, match="condenser_model"):
        chain_module._build_chain(_settings(condenser_model=""))


def test_build_chain_raises_when_condenser_provider_missing() -> None:
    with pytest.raises(RuntimeError, match="condenser_provider"):
        chain_module._build_chain(_settings(condenser_provider=""))


def test_build_chain_raises_when_api_key_missing() -> None:
    with pytest.raises(RuntimeError, match="openrouter_api_key"):
        chain_module._build_chain(_settings(openrouter_api_key=""))


# --- _build_chain: the pinned-routing wiring (SPEC §8.2, §8.3) ---------------


def test_build_chain_pins_model_key_and_base_url() -> None:
    chain_module._build_chain(_settings())

    [client] = _FakeChatOpenAI.instances
    assert client.kwargs["model"] == "mistralai/mistral-small-3.2-24b-instruct"
    assert client.kwargs["base_url"] == OPENROUTER_BASE_URL
    assert client.kwargs["api_key"].get_secret_value() == "sk-or-test"


def test_build_chain_pins_provider_routing() -> None:
    chain_module._build_chain(_settings(condenser_provider="mistral/eu"))

    [client] = _FakeChatOpenAI.instances
    assert client.kwargs["extra_body"] == {
        "provider": {
            "require_parameters": True,
            "allow_fallbacks": False,
            "order": ["mistral/eu"],
        }
    }


def test_build_chain_uses_condenser_output_directly_not_a_wrapper() -> None:
    """Unlike generation's `_EnvelopeContainer` (needed because `Envelope` is a root-level
    union), `CondenserOutput` is already a plain object — it goes to
    `with_structured_output` unwrapped."""
    chain_module._build_chain(_settings())

    [client] = _FakeChatOpenAI.instances
    assert client.structured_schema is CondenserOutput
    assert client.structured_strict is True


# --- _cached_chain: one client per distinct Settings -------------------------


def test_cached_chain_builds_once_for_the_same_settings() -> None:
    settings = _settings()

    chain_module._cached_chain(settings)
    chain_module._cached_chain(settings)

    assert len(_FakeChatOpenAI.instances) == 1


def test_cached_chain_rebuilds_for_different_settings() -> None:
    chain_module._cached_chain(_settings(condenser_provider="mistral/eu"))
    chain_module._cached_chain(_settings(condenser_provider="mistral"))

    assert len(_FakeChatOpenAI.instances) == 2


# --- make_condense_fn: the seam pipeline.condense() actually calls -----------


class _FakeInvokeChain:
    def __init__(self, result: CondenserOutput | Exception) -> None:
        self._result = result
        self.received: list[Any] | None = None

    def invoke(self, messages: list[Any]) -> CondenserOutput:
        self.received = messages
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def test_condense_fn_returns_the_chains_output_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    output = CondenserOutput(requete="Une question autonome ?")
    fake_chain = _FakeInvokeChain(output)
    monkeypatch.setattr(chain_module, "_build_chain", lambda settings: fake_chain)

    condense_fn = chain_module.make_condense_fn(_settings())
    messages: list[Message] = [("system", "instructions"), ("user", "question")]

    result = condense_fn(messages)

    assert result is output
    assert fake_chain.received is not None
    assert [type(m).__name__ for m in fake_chain.received] == ["SystemMessage", "HumanMessage"]


def test_condense_fn_propagates_call_failures_rather_than_swallowing_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`condense()` (`pipeline.py`) is what turns a call failure into
    `CondenseStatus.FALLBACK_ERROR` — this seam must not catch it first."""
    fake_chain = _FakeInvokeChain(TimeoutError("pinned endpoint down"))
    monkeypatch.setattr(chain_module, "_build_chain", lambda settings: fake_chain)

    condense_fn = chain_module.make_condense_fn(_settings())

    with pytest.raises(TimeoutError, match="pinned endpoint down"):
        condense_fn([("user", "question")])
