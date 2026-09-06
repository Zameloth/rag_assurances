"""The real `GenerateFn` seam (`rag.generation.chain`, SPEC §10.1, #42), against a fake
`ChatOpenAI`/`with_structured_output` chain.

Never calls OpenRouter: the contract under test is message conversion, the pinned-routing
`extra_body` shape, per-`Settings` chain caching and the `_EnvelopeContainer` unwrap — not
the model's own answer quality, which only a live call (this repo's
`scripts`-adjacent smoke test, not the suite) can speak to.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import rag.generation.chain as chain_module
from rag.config import Settings
from rag.generation.chain import OPENROUTER_BASE_URL, _EnvelopeContainer, _to_langchain_messages
from rag.generation.prompt import Message
from rag.generation.schema import Motif, Refus, Reponse


def _settings(**overrides: Any) -> Settings:
    fields: dict[str, Any] = {
        "openrouter_api_key": "sk-or-test",
        "generation_model": "mistralai/mistral-large-2512",
        "generation_provider": "mistral/eu",
        "condenser_model": "",
        "condenser_provider": "",
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
    `lru_cache` singleton on both sides of the test — a leaked cache would otherwise let
    one test's chain answer another's, exactly the risk `test_embedder.py` guards against
    for BGE-M3."""
    _FakeChatOpenAI.instances.clear()
    monkeypatch.setattr(chain_module, "ChatOpenAI", _FakeChatOpenAI)
    chain_module._cached_chain.cache_clear()
    yield
    chain_module._cached_chain.cache_clear()


# --- _to_langchain_messages --------------------------------------------------


def test_to_langchain_messages_maps_roles_in_order() -> None:
    messages: list[Message] = [
        ("system", "instructions"),
        ("user", "question"),
        ("assistant", "réponse précédente"),
    ]

    result = _to_langchain_messages(messages)

    assert [type(m).__name__ for m in result] == ["SystemMessage", "HumanMessage", "AIMessage"]
    assert [m.content for m in result] == ["instructions", "question", "réponse précédente"]


# --- _build_chain: the consumer-side checks (Settings §16.3's "each consumer checks what
# it uses") -------------------------------------------------------------------


def test_build_chain_raises_when_generation_model_missing() -> None:
    with pytest.raises(RuntimeError, match="generation_model"):
        chain_module._build_chain(_settings(generation_model=""))


def test_build_chain_raises_when_generation_provider_missing() -> None:
    with pytest.raises(RuntimeError, match="generation_provider"):
        chain_module._build_chain(_settings(generation_provider=""))


def test_build_chain_raises_when_api_key_missing() -> None:
    with pytest.raises(RuntimeError, match="openrouter_api_key"):
        chain_module._build_chain(_settings(openrouter_api_key=""))


# --- _build_chain: the pinned-routing wiring (SPEC §10.1) --------------------


def test_build_chain_pins_model_key_and_base_url() -> None:
    chain_module._build_chain(_settings())

    [client] = _FakeChatOpenAI.instances
    assert client.kwargs["model"] == "mistralai/mistral-large-2512"
    assert client.kwargs["base_url"] == OPENROUTER_BASE_URL
    assert client.kwargs["api_key"].get_secret_value() == "sk-or-test"


def test_build_chain_pins_provider_routing_per_spec_10_1() -> None:
    chain_module._build_chain(_settings(generation_provider="mistral/eu"))

    [client] = _FakeChatOpenAI.instances
    assert client.kwargs["extra_body"] == {
        "provider": {
            "require_parameters": True,
            "allow_fallbacks": False,
            "order": ["mistral/eu"],
        }
    }


def test_build_chain_uses_the_envelope_container_not_envelope_directly() -> None:
    """OpenAI's `strict: true` rejects an `anyOf` at the schema root, and `Envelope`
    (`schema.py`) is exactly that — the schema handed to `with_structured_output` must be
    `_EnvelopeContainer`, whose root is a plain object with `envelope` as an inner
    property."""
    chain_module._build_chain(_settings())

    [client] = _FakeChatOpenAI.instances
    assert client.structured_schema is _EnvelopeContainer
    assert client.structured_strict is True


# --- _cached_chain: one client per distinct Settings -------------------------


def test_cached_chain_builds_once_for_the_same_settings() -> None:
    settings = _settings()

    chain_module._cached_chain(settings)
    chain_module._cached_chain(settings)

    assert len(_FakeChatOpenAI.instances) == 1


def test_cached_chain_rebuilds_for_different_settings() -> None:
    chain_module._cached_chain(_settings(generation_provider="mistral/eu"))
    chain_module._cached_chain(_settings(generation_provider="mistral"))

    assert len(_FakeChatOpenAI.instances) == 2


# --- make_generate_fn: the seam pipeline.generate() actually calls -----------


class _FakeInvokeChain:
    def __init__(self, result: _EnvelopeContainer) -> None:
        self._result = result
        self.received: list[Any] | None = None

    def invoke(self, messages: list[Any]) -> _EnvelopeContainer:
        self.received = messages
        return self._result


def test_generate_fn_unwraps_a_reponse_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    reponse = Reponse(explanation="texte")
    fake_chain = _FakeInvokeChain(_EnvelopeContainer(envelope=reponse))
    monkeypatch.setattr(chain_module, "_build_chain", lambda settings: fake_chain)

    generate_fn = chain_module.make_generate_fn(_settings())
    messages: list[Message] = [("system", "instructions"), ("user", "question")]

    result = generate_fn(messages)

    assert isinstance(result, Reponse)
    assert result.explanation == "texte"
    assert fake_chain.received is not None
    assert [type(m).__name__ for m in fake_chain.received] == ["SystemMessage", "HumanMessage"]


def test_generate_fn_unwraps_a_refus_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    refus = Refus(explanation="texte", motif=Motif.HORS_CORPUS)
    fake_chain = _FakeInvokeChain(_EnvelopeContainer(envelope=refus))
    monkeypatch.setattr(chain_module, "_build_chain", lambda settings: fake_chain)

    generate_fn = chain_module.make_generate_fn(_settings())

    result = generate_fn([("user", "question")])

    assert isinstance(result, Refus)
    assert result.motif is Motif.HORS_CORPUS


def test_generate_fn_never_leaks_the_envelope_container(monkeypatch: pytest.MonkeyPatch) -> None:
    """SPEC §10's typed envelope is `Reponse | Refus` — `_EnvelopeContainer` is a wire-format
    artefact of `chain.py` alone and must never reach the `GenerateFn` caller."""
    fake_chain = _FakeInvokeChain(_EnvelopeContainer(envelope=Reponse(explanation="texte")))
    monkeypatch.setattr(chain_module, "_build_chain", lambda settings: fake_chain)

    generate_fn = chain_module.make_generate_fn(_settings())

    result = generate_fn([("user", "question")])

    assert not isinstance(result, _EnvelopeContainer)
