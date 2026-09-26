#!/usr/bin/env python3
"""Create or update the Langfuse LLM connection to OpenRouter (SPEC §11.1, ADR-0025, #47).

OpenRouter is configured as provider **OpenAI** with the gateway URL as base URL — the
gateway must do tool calling in OpenAI format, because managed judges extract `score` and
`reasoning` via a function call. Only `JUDGE_MODEL` is exposed on the connection.

This connection is what the Langfuse UI's managed evaluators (Faithfulness v2 included)
run on. The eval harness's own judge does **not** go through it: a Langfuse connection can't
pin OpenRouter's provider routing or report the resolved provider back, so eval runs judge
client-side (`rag.eval.judge_chain.make_judge`). Idempotent: re-running upserts the same
connection.

    uv run python scripts/configure_langfuse_judge_connection.py

Sends `OPENROUTER_API_KEY` to the Langfuse project configured by `LANGFUSE_*`.
"""

from __future__ import annotations

from langfuse import Langfuse

from rag.config import load_settings
from rag.eval.judge_chain import judge_llm_connection


def main() -> int:
    settings = load_settings()
    connection = judge_llm_connection(settings)
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )
    result = langfuse.api.llm_connections.upsert(**connection)
    print(
        f"Langfuse LLM connection {result.provider!r} ({result.adapter}) -> {result.base_url}, "
        f"models: {', '.join(result.custom_models)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
