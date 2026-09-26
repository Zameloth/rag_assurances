"""The real `JudgeCall` (`rag.eval.judge_chain`, SPEC §12.10, #47), against fakes — pinned
routing, the resolved provider read back off the response, the same-family guard. Never
calls OpenRouter."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langfuse.api import LlmAdapter

from rag.config import ConfigurationError, Settings
from rag.eval.judge import FaithfulnessOutput, JudgeOutputError, PromptLanguage
from rag.eval.judge_chain import (
    ProviderReportingChatOpenAI,
    _call_with,
    _judge_llm,
    check_judge_family,
    judge_llm_connection,
    make_judge,
)
from rag.generation.chain import OPENROUTER_BASE_URL


def _settings(**overrides: Any) -> Settings:
    fields: dict[str, Any] = {
        "openrouter_api_key": "sk-or-test",
        "generation_model": "mistralai/mistral-large-2512",
        "generation_provider": "mistral/eu",
        "condenser_model": "",
        "condenser_provider": "",
        "judge_model": "anthropic/claude-sonnet-5",
        "judge_provider": "anthropic",
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
        "langfuse_base_url": "https://cloud.langfuse.com",
        "langfuse_tracing": False,
        "qdrant_url": "http://localhost:6333",
    }
    fields.update(overrides)
    return Settings(**fields)


def _openrouter_response(provider: str | None) -> dict[str, Any]:
    response: dict[str, Any] = {
        "id": "gen-1",
        "model": "anthropic/claude-sonnet-5",
        "object": "chat.completion",
        "created": 0,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    if provider is not None:
        response["provider"] = provider
    return response


class TestProviderReportingChatOpenAI:
    """`ChatOpenAI` builds `response_metadata` from a fixed key allowlist and drops
    OpenRouter's `provider` field (`rag.eval.run_generation_experiment`'s docstring) —
    this subclass exists only to keep it."""

    def _llm(self) -> ProviderReportingChatOpenAI:
        return ProviderReportingChatOpenAI(api_key="sk", model="m", base_url=OPENROUTER_BASE_URL)  # type: ignore[arg-type]

    def test_resolved_provider_reaches_response_metadata(self) -> None:
        result = self._llm()._create_chat_result(_openrouter_response("Google"))

        assert result.generations[0].message.response_metadata["provider"] == "Google"

    def test_absent_provider_stays_absent(self) -> None:
        result = self._llm()._create_chat_result(_openrouter_response(None))

        assert "provider" not in result.generations[0].message.response_metadata


class TestJudgeLlm:
    def test_pins_routing_with_no_fallbacks(self) -> None:
        llm = _judge_llm(_settings())

        assert isinstance(llm, ProviderReportingChatOpenAI)
        assert llm.model_name == "anthropic/claude-sonnet-5"
        assert llm.openai_api_base == OPENROUTER_BASE_URL
        assert llm.extra_body == {
            "provider": {"require_parameters": True, "allow_fallbacks": False, "order": ["anthropic"]}
        }

    def test_is_deterministic_as_far_as_the_api_allows(self) -> None:
        assert _judge_llm(_settings()).temperature == 0

    @pytest.mark.parametrize("missing", ["judge_model", "judge_provider", "openrouter_api_key"])
    def test_refuses_to_build_without_what_it_pins(self, missing: str) -> None:
        with pytest.raises(RuntimeError, match=missing):
            _judge_llm(_settings(**{missing: ""}))


class _FakeStructured:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.invoked_with: list[BaseMessage] | None = None

    def invoke(self, messages: list[BaseMessage]) -> dict[str, Any]:
        self.invoked_with = messages
        return self.result


class _FakeLlm:
    def __init__(self, result: dict[str, Any]) -> None:
        self.structured = _FakeStructured(result)
        self.structured_kwargs: dict[str, Any] = {}

    def with_structured_output(self, schema: type, **kwargs: Any) -> _FakeStructured:
        self.structured_kwargs = {"schema": schema, **kwargs}
        return self.structured


class TestCallWith:
    def _raw(self, provider: str | None) -> AIMessage:
        metadata = {"provider": provider} if provider is not None else {}
        return AIMessage(content="", response_metadata=metadata)

    def test_returns_parsed_output_and_resolved_provider(self) -> None:
        parsed = FaithfulnessOutput(score=1.0, reasoning="ok")
        llm = _FakeLlm({"raw": self._raw("Anthropic"), "parsed": parsed, "parsing_error": None})

        output, provider = _call_with(cast(ChatOpenAI, llm))("the prompt", FaithfulnessOutput)

        assert output is parsed
        assert provider == "Anthropic"
        assert llm.structured.invoked_with == [HumanMessage("the prompt")]

    def test_extracts_through_a_function_call_like_managed_judges(self) -> None:
        llm = _FakeLlm({"raw": self._raw("Anthropic"), "parsed": FaithfulnessOutput(score=1, reasoning=""), "parsing_error": None})

        _call_with(cast(ChatOpenAI, llm))("p", FaithfulnessOutput)

        assert llm.structured_kwargs == {"schema": FaithfulnessOutput, "method": "function_calling", "include_raw": True}

    def test_a_parsing_error_raises_rather_than_scoring(self) -> None:
        llm = _FakeLlm({"raw": self._raw("Anthropic"), "parsed": None, "parsing_error": ValueError("bad json")})

        with pytest.raises(JudgeOutputError, match="bad json"):
            _call_with(cast(ChatOpenAI, llm))("p", FaithfulnessOutput)

    def test_an_unreported_provider_raises_rather_than_pinning_nothing(self) -> None:
        """SPEC §12.10: the resolved provider is recorded in *every* persisted run."""
        llm = _FakeLlm({"raw": self._raw(None), "parsed": FaithfulnessOutput(score=1, reasoning=""), "parsing_error": None})

        with pytest.raises(JudgeOutputError, match="provider"):
            _call_with(cast(ChatOpenAI, llm))("p", FaithfulnessOutput)


class TestJudgeFamily:
    def test_a_different_family_passes(self) -> None:
        check_judge_family("anthropic/claude-sonnet-5", "mistralai/mistral-large-2512")

    def test_the_same_family_is_rejected(self) -> None:
        """SPEC §12.10 — a judge grading its own family carries self-preference bias."""
        with pytest.raises(ConfigurationError, match="same family"):
            check_judge_family("mistralai/mistral-medium", "mistralai/mistral-large-2512")


class TestMakeJudge:
    def test_carries_the_model_and_both_configured_languages(self) -> None:
        judge = make_judge(_settings(judge_faithfulness_language="fr", judge_point_coverage_language="en"))

        assert judge.model == "anthropic/claude-sonnet-5"
        assert judge.faithfulness_language is PromptLanguage.FR
        assert judge.point_coverage_language is PromptLanguage.EN

    def test_rejects_a_same_family_judge_before_any_call(self) -> None:
        with pytest.raises(ConfigurationError):
            make_judge(_settings(judge_model="mistralai/mistral-medium"))


class TestLangfuseLlmConnection:
    """SPEC §11.1 — OpenRouter reaches Langfuse as provider **OpenAI** with the gateway URL
    as base URL, since managed judges extract `score`/`reasoning` via an OpenAI-format
    function call."""

    def test_uses_the_openai_adapter_pointed_at_the_gateway(self) -> None:
        connection = judge_llm_connection(_settings())

        assert connection["adapter"] is LlmAdapter.OPEN_AI
        assert connection["base_url"] == OPENROUTER_BASE_URL
        assert connection["secret_key"] == "sk-or-test"

    def test_exposes_only_the_judge_model(self) -> None:
        connection = judge_llm_connection(_settings())

        assert connection["custom_models"] == ["anthropic/claude-sonnet-5"]
        assert connection["with_default_models"] is False

    def test_is_named_for_the_gateway_so_re_running_upserts_rather_than_duplicates(self) -> None:
        assert judge_llm_connection(_settings())["provider"] == "openrouter"

    def test_refuses_without_a_key_or_a_judge_model(self) -> None:
        with pytest.raises(RuntimeError):
            judge_llm_connection(_settings(openrouter_api_key=""))
        with pytest.raises(RuntimeError):
            judge_llm_connection(_settings(judge_model=""))
