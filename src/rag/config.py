"""Configuration, read from the environment and nowhere else (SPEC §16.3).

`.env` at the repo root is the developer's copy and is gitignored; `.env.example` is the
committed template and the record of what every variable means. Nothing in this package
carries a model id, a key or a URL inline — every one of them arrives through `Settings`.

**The model ids have no defaults on purpose.** `GENERATION_MODEL` is the ablatable arm and
the other two are the controls; a source-level fallback would let a run whose `.env` is
missing or misspelled resolve to a model nobody chose, and "which arm did this rung
actually run" is the one question the ladder cannot afford to answer wrongly. Absent means
empty here, and the consumer that reads a value it needs is the one that fails on it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

__all__ = ["ConfigurationError", "Settings", "load_settings"]

# The two defaults that are spec constants rather than choices: §11.3 fixes the Langfuse
# URL, and §16.1's dev compose fixes where Qdrant answers. Neither identifies a run.
DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
DEFAULT_QDRANT_URL = "http://localhost:6333"

TRUTHY = frozenset({"true", "1", "yes", "on"})
FALSY = frozenset({"false", "0", "no", "off", ""})


class ConfigurationError(Exception):
    """A variable is present but its value cannot be honoured."""


@dataclass(frozen=True)
class Settings:
    """The SPEC §16.3 table, resolved.

    Absent variables read as empty strings rather than raising: the pipeline stages have
    disjoint needs — ingest talks to Qdrant and no LLM, the app never reads `JUDGE_MODEL` —
    so a single up-front gate would make every stage hold credentials for roles it does not
    play. Each consumer checks what it uses.
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
        """Resolve settings from a mapping.

        Raises `ConfigurationError` for a value that is present but unusable — a variable
        that cannot mean what it says is a different thing from one that is simply unset.
        """
        _reject_dead_sdk_v3_name(env)

        return cls(
            openrouter_api_key=env.get("OPENROUTER_API_KEY", ""),
            generation_model=env.get("GENERATION_MODEL", ""),
            condenser_model=env.get("CONDENSER_MODEL", ""),
            judge_model=env.get("JUDGE_MODEL", ""),
            langfuse_public_key=env.get("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=env.get("LANGFUSE_SECRET_KEY", ""),
            langfuse_base_url=env.get("LANGFUSE_BASE_URL") or DEFAULT_LANGFUSE_BASE_URL,
            langfuse_tracing=_parse_bool(env.get("LANGFUSE_TRACING"), name="LANGFUSE_TRACING"),
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
    """SPEC §11.2 — unset means off, and a typo means nothing at all."""
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
