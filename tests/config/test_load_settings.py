"""SPEC §16.3 — `.env` at the repo root, loaded via python-dotenv."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from rag.config import load_settings

SETTING_NAMES = (
    "OPENROUTER_API_KEY",
    "GENERATION_MODEL",
    "CONDENSER_MODEL",
    "JUDGE_MODEL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_TRACING",
    "LANGFUSE_HOST",
    "QDRANT_URL",
)


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    """Fence these tests off from the process environment, in both directions.

    Inward, so the developer's own `.env` cannot satisfy an assertion about a missing
    variable. Outward, because `load_dotenv` writes into `os.environ` and monkeypatch
    cannot undo a name it never saw — without the restore, a fixture value set here
    leaks into every test that runs afterwards.
    """
    saved = os.environ.copy()
    for name in SETTING_NAMES:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def write_env(tmp_path: Path, body: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(body, encoding="utf-8")
    return env_file


def test_reads_values_out_of_the_env_file(tmp_path: Path) -> None:
    env_file = write_env(
        tmp_path,
        "OPENROUTER_API_KEY=sk-or-from-file\nQDRANT_URL=http://qdrant:6333\n",
    )

    settings = load_settings(env_file)

    assert settings.openrouter_api_key == "sk-or-from-file"
    assert settings.qdrant_url == "http://qdrant:6333"


def test_the_real_environment_wins_over_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = write_env(tmp_path, "OPENROUTER_API_KEY=sk-or-from-file\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-shell")

    assert load_settings(env_file).openrouter_api_key == "sk-or-from-shell"


def test_a_missing_env_file_is_not_an_error_when_the_shell_has_the_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-shell")

    settings = load_settings(tmp_path / "absent.env")

    assert settings.openrouter_api_key == "sk-or-from-shell"
    assert settings.langfuse_tracing is False
