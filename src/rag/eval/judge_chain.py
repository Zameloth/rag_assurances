"""The real `JudgeCall`: `ChatOpenAI` on OpenRouter with pinned routing, reading back the
provider OpenRouter actually resolved (SPEC §10.1, §12.10, ADR-0025, #47).

Sibling to `rag.generation.chain`, same split: `rag.eval.judge` declares the seam and stays
free of any LLM SDK, this module is the one that imports `langchain_openai`.

**Pinned routing, as for generation** — `require_parameters`, `allow_fallbacks: false`, a
one-element `order` — because OpenRouter routing is a variance source *inside* the eval
harness: a judge whose backing endpoint moves between two runs reintroduces on the scoring
side exactly the drift manual annotation was paid to remove on the label side (SPEC §12.10).

**The resolved provider is read off the response, not assumed from the request.**
`ChatOpenAI._create_chat_result` builds `response_metadata` from a fixed key allowlist that
drops OpenRouter's top-level `provider` field (the gap `rag.eval.run_generation_experiment`
documents for the generation call). `ProviderReportingChatOpenAI` puts it back, and
`with_structured_output(include_raw=True)` hands the raw message through, so every judge
score carries the provider that really served it. A response without one raises rather than
pinning an empty string.

**Function calling, like Langfuse's managed judges** (SPEC §11.1: they "extract `score` and
`reasoning` via a function call") — the same extraction path the managed Faithfulness v2
template was written for, so running it client-side changes where it runs, not how its
answer is read.
"""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from langfuse.api import LlmAdapter
from pydantic import BaseModel, SecretStr

from rag.config import ConfigurationError, Settings
from rag.eval.judge import Judge, JudgeCall, JudgeOutputError
from rag.generation.chain import OPENROUTER_BASE_URL

__all__ = [
    "LANGFUSE_CONNECTION_PROVIDER",
    "ProviderReportingChatOpenAI",
    "check_judge_family",
    "judge_llm_connection",
    "make_judge",
]

# The connection's name in Langfuse — its upsert key, so re-running the setup script
# updates this connection rather than adding a second one.
LANGFUSE_CONNECTION_PROVIDER = "openrouter"

_M = TypeVar("_M", bound=BaseModel)


class ProviderReportingChatOpenAI(ChatOpenAI):
    """`ChatOpenAI` that keeps OpenRouter's `provider` response field in
    `response_metadata` — the only change."""

    def _create_chat_result(self, response: Any, generation_info: dict[str, Any] | None = None) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()
        provider = response_dict.get("provider")
        if provider:
            for generation in result.generations:
                generation.message.response_metadata["provider"] = provider
        return result


def check_judge_family(judge_model: str, generation_model: str) -> None:
    """SPEC §12.10's binding requirement: the judge is a different family from the
    generation arm — a judge grading its own family carries self-preference bias toward it.
    Family is the OpenRouter vendor prefix (`mistralai/…`, `anthropic/…`)."""
    judge_family = judge_model.split("/", 1)[0]
    if judge_family == generation_model.split("/", 1)[0]:
        raise ConfigurationError(
            f"JUDGE_MODEL={judge_model!r} is the same family as GENERATION_MODEL={generation_model!r}; "
            "the judge must be a different family from every generation arm (SPEC §12.10)."
        )


def _judge_llm(settings: Settings) -> ProviderReportingChatOpenAI:
    if not settings.judge_model:
        raise RuntimeError("judge_model not provided")
    if not settings.judge_provider:
        raise RuntimeError("judge_provider not provided")
    if not settings.openrouter_api_key:
        raise RuntimeError("openrouter_api_key not provided")
    return ProviderReportingChatOpenAI(
        api_key=SecretStr(settings.openrouter_api_key),
        model=settings.judge_model,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        extra_body={
            "provider": {
                "require_parameters": True,
                "allow_fallbacks": False,
                "order": [settings.judge_provider],
            }
        },
    )


def _call_with(llm: ChatOpenAI) -> JudgeCall:
    def call(prompt: str, schema: type[_M]) -> tuple[_M, str]:
        structured = llm.with_structured_output(schema, method="function_calling", include_raw=True)
        result = structured.invoke([HumanMessage(prompt)])
        assert isinstance(result, dict)
        if result["parsing_error"] is not None:
            raise JudgeOutputError(f"judge output did not parse: {result['parsing_error']}")
        parsed = result["parsed"]
        if not isinstance(parsed, schema):
            raise JudgeOutputError(f"judge returned no {schema.__name__}")
        raw = result["raw"]
        provider = raw.response_metadata.get("provider") if isinstance(raw, AIMessage) else None
        if not provider:
            raise JudgeOutputError("OpenRouter reported no resolved provider for the judge call")
        return parsed, str(provider)

    return call


def make_judge(settings: Settings) -> Judge:
    """The configured judge: `JUDGE_MODEL` on its pinned provider, prompt language per
    metric from `JUDGE_*_LANGUAGE`. Rejects a same-family judge before any call is made."""
    check_judge_family(settings.judge_model, settings.generation_model)
    return Judge(
        call=_call_with(_judge_llm(settings)),
        model=settings.judge_model,
        faithfulness_language=settings.judge_faithfulness_language,
        point_coverage_language=settings.judge_point_coverage_language,
    )


def judge_llm_connection(settings: Settings) -> dict[str, Any]:
    """Keyword arguments for Langfuse's `llm_connections.upsert` — OpenRouter as an
    **OpenAI**-adapter connection with the gateway as base URL (SPEC §11.1: managed judges
    extract `score` and `reasoning` via an OpenAI-format function call).

    This is what makes the managed Faithfulness v2 evaluator runnable from the Langfuse UI
    (e.g. as a live evaluator on production traces). It is not what scores eval runs:
    a connection has no way to carry OpenRouter's `provider` routing block or to report the
    resolved provider back, so the pinned, provider-recorded judge is `make_judge`'s
    client-side one (ADR-0025). Only `JUDGE_MODEL` is exposed on it, so a UI evaluator
    can't silently pick a same-family model.
    """
    if not settings.openrouter_api_key:
        raise RuntimeError("openrouter_api_key not provided")
    if not settings.judge_model:
        raise RuntimeError("judge_model not provided")
    return {
        "provider": LANGFUSE_CONNECTION_PROVIDER,
        "adapter": LlmAdapter.OPEN_AI,
        "secret_key": settings.openrouter_api_key,
        "base_url": OPENROUTER_BASE_URL,
        "custom_models": [settings.judge_model],
        "with_default_models": False,
    }
