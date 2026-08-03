"""Configuration, read from the environment and nowhere else (SPEC §16.3).

`.env` at the repo root is the developer's copy and is gitignored; `.env.example` is the
committed template. Nothing in this package reads a model id, a key or a URL inline —
every one of them arrives through `Settings`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

__all__ = [
    "ConfigurationError",
    "MissingSettingError",
    "Settings",
    "load_settings",
]

DEFAULT_GENERATION_MODEL = "mistralai/mistral-large-2512"
DEFAULT_CONDENSER_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
DEFAULT_QDRANT_URL = "http://localhost:6333"

TRUTHY = frozenset({"true", "1", "yes", "on"})
FALSY = frozenset({"false", "0", "no", "off", ""})


class ConfigurationError(Exception):
    """A variable is present but its value cannot be honoured."""


class MissingSettingError(ConfigurationError):
    """A variable the run needs has no value anywhere."""


@dataclass(frozen=True)
class Settings:
    """The §16.3 table, resolved.

    Construct with `from_env` rather than by hand outside tests: the validation the
    constructor skips is the whole point of the class.
    """

    openrouter_api_key: str
    generation_model: str
    condenser_model: str
    judge_model: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str
    langfuse_tracing: bool
    qdrant_url: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        """Resolve settings from a mapping, reporting every problem it finds at once.

        Raises `MissingSettingError` when a needed variable is absent and
        `ConfigurationError` when one is present but unusable.
        """
        _reject_dead_sdk_v3_name(env)

        tracing = _parse_bool(env.get("LANGFUSE_TRACING"), name="LANGFUSE_TRACING")
        generation_model = env.get("GENERATION_MODEL") or DEFAULT_GENERATION_MODEL
        judge_model = env.get("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL

        # Langfuse keys are needed only where tracing is on, so a dev running under
        # §11.2's default never has to hold credentials for a service it is not calling.
        required = ["OPENROUTER_API_KEY"]
        if tracing:
            required += ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        missing = [name for name in required if not env.get(name)]
        if missing:
            raise MissingSettingError(
                "Missing required configuration: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill it in."
            )

        _reject_same_family_judge(generation_model, judge_model)

        return cls(
            openrouter_api_key=env["OPENROUTER_API_KEY"],
            generation_model=generation_model,
            condenser_model=env.get("CONDENSER_MODEL") or DEFAULT_CONDENSER_MODEL,
            judge_model=judge_model,
            langfuse_public_key=env.get("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=env.get("LANGFUSE_SECRET_KEY", ""),
            langfuse_base_url=env.get("LANGFUSE_BASE_URL") or DEFAULT_LANGFUSE_BASE_URL,
            langfuse_tracing=tracing,
            qdrant_url=env.get("QDRANT_URL") or DEFAULT_QDRANT_URL,
        )


def load_settings(env_file: Path | str | None = None) -> Settings:
    """Load `.env` and resolve the settings from the process environment.

    A real environment variable wins over the file, so a one-off `LANGFUSE_TRACING=true`
    in front of a command works without editing anything. An absent file is not an
    error — CI and the container both pass their values in directly.
    """
    load_dotenv(env_file)
    return Settings.from_env(os.environ)


def _parse_bool(raw: str | None, *, name: str) -> bool:
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in TRUTHY:
        return True
    if value in FALSY:
        return False
    # A typo must not read as off: §11.2 makes tracing a gate, and a silently disabled
    # gate is discovered only by the missing traces you went looking for.
    raise ConfigurationError(
        f"{name}={raw!r} is neither true nor false. Use one of: "
        + ", ".join(sorted(TRUTHY | (FALSY - {""})))
    )


def _reject_dead_sdk_v3_name(env: Mapping[str, str]) -> None:
    """SPEC §11.3 — `LANGFUSE_HOST` is the SDK v3 name and v4 ignores it in silence.

    On EU cloud the fallback default happens to be right, so the mistake surfaces only
    once someone configures a US or self-hosted URL and wonders why it had no effect.
    """
    if env.get("LANGFUSE_HOST"):
        raise ConfigurationError(
            "LANGFUSE_HOST is the SDK v3 name and is ignored by langfuse>=4; "
            "rename it to LANGFUSE_BASE_URL."
        )


def _reject_same_family_judge(generation_model: str, judge_model: str) -> None:
    """SPEC §12.10 — the judge must not grade its own family.

    The generation model is an ablatable arm, so this is checked per run rather than
    settled once: a rung that swaps in the judge's family invalidates its own scores
    through known self-preference bias.
    """
    if _family(judge_model) == _family(generation_model):
        raise ConfigurationError(
            f"JUDGE_MODEL={judge_model!r} shares a family with "
            f"GENERATION_MODEL={generation_model!r}; the judge must come from another family."
        )


def _family(model_id: str) -> str:
    """The vendor prefix of an OpenRouter model id — `mistralai/mistral-large` → `mistralai`."""
    return model_id.split("/", 1)[0].lower()
