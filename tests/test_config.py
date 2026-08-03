"""SPEC §16.3 — every value the pipeline runs on comes from the environment."""

import pytest

from rag.config import ConfigurationError, MissingSettingError, Settings

MINIMAL = {"OPENROUTER_API_KEY": "sk-or-test"}


def test_reads_every_variable_the_spec_table_lists() -> None:
    settings = Settings.from_env(
        {
            "OPENROUTER_API_KEY": "sk-or-test",
            "GENERATION_MODEL": "arm/under-test",
            "CONDENSER_MODEL": "mistralai/mistral-small-3.2-24b-instruct",
            "JUDGE_MODEL": "anthropic/claude-sonnet-5",
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
            "LANGFUSE_BASE_URL": "https://cloud.langfuse.com",
            "LANGFUSE_TRACING": "true",
            "QDRANT_URL": "http://qdrant:6333",
        }
    )

    assert settings.openrouter_api_key == "sk-or-test"
    assert settings.generation_model == "arm/under-test"
    assert settings.condenser_model == "mistralai/mistral-small-3.2-24b-instruct"
    assert settings.judge_model == "anthropic/claude-sonnet-5"
    assert settings.langfuse_public_key == "pk-lf-test"
    assert settings.langfuse_secret_key == "sk-lf-test"
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"
    assert settings.langfuse_tracing is True
    assert settings.qdrant_url == "http://qdrant:6333"


def test_model_defaults_match_the_spec() -> None:
    settings = Settings.from_env(MINIMAL)

    assert settings.generation_model == "mistralai/mistral-large-2512"
    assert settings.condenser_model == "mistralai/mistral-small-3.2-24b-instruct"
    assert settings.judge_model == "anthropic/claude-sonnet-5"
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"
    assert settings.qdrant_url == "http://localhost:6333"


class TestLangfuseTracing:
    """SPEC §11.2 — the free tier fails on debugging, so dev traces nothing."""

    def test_defaults_to_false_when_unset(self) -> None:
        assert Settings.from_env(MINIMAL).langfuse_tracing is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "on"])
    def test_accepts_the_usual_truthy_spellings(self, value: str) -> None:
        env = MINIMAL | {
            "LANGFUSE_TRACING": value,
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
            "LANGFUSE_SECRET_KEY": "sk-lf-test",
        }
        assert Settings.from_env(env).langfuse_tracing is True

    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off", ""])
    def test_accepts_the_usual_falsy_spellings(self, value: str) -> None:
        assert Settings.from_env(MINIMAL | {"LANGFUSE_TRACING": value}).langfuse_tracing is False

    def test_rejects_a_typo_rather_than_silently_reading_it_as_off(self) -> None:
        with pytest.raises(ConfigurationError, match="LANGFUSE_TRACING"):
            Settings.from_env(MINIMAL | {"LANGFUSE_TRACING": "ture"})


class TestRequiredSettings:
    def test_openrouter_key_is_always_required(self) -> None:
        with pytest.raises(MissingSettingError, match="OPENROUTER_API_KEY"):
            Settings.from_env({})

    def test_langfuse_keys_are_not_required_while_tracing_is_off(self) -> None:
        settings = Settings.from_env(MINIMAL)

        assert settings.langfuse_public_key == ""
        assert settings.langfuse_secret_key == ""

    def test_langfuse_keys_become_required_once_tracing_is_on(self) -> None:
        with pytest.raises(MissingSettingError) as excinfo:
            Settings.from_env(MINIMAL | {"LANGFUSE_TRACING": "true"})

        assert "LANGFUSE_PUBLIC_KEY" in str(excinfo.value)
        assert "LANGFUSE_SECRET_KEY" in str(excinfo.value)

    def test_names_every_missing_variable_at_once(self) -> None:
        with pytest.raises(MissingSettingError) as excinfo:
            Settings.from_env({"LANGFUSE_TRACING": "true"})

        message = str(excinfo.value)
        assert "OPENROUTER_API_KEY" in message
        assert "LANGFUSE_PUBLIC_KEY" in message


class TestVersionTraps:
    """SPEC §11.3 and §12.10 — two settings whose wrong value fails silently."""

    def test_langfuse_host_is_rejected_as_the_dead_sdk_v3_name(self) -> None:
        with pytest.raises(ConfigurationError, match="LANGFUSE_BASE_URL"):
            Settings.from_env(MINIMAL | {"LANGFUSE_HOST": "https://cloud.langfuse.com"})

    def test_judge_may_not_share_a_family_with_the_generation_arm(self) -> None:
        env = MINIMAL | {
            "GENERATION_MODEL": "mistralai/mistral-large-2512",
            "JUDGE_MODEL": "mistralai/mistral-medium-3",
        }

        with pytest.raises(ConfigurationError, match="family"):
            Settings.from_env(env)

    def test_a_judge_from_another_family_is_accepted(self) -> None:
        env = MINIMAL | {
            "GENERATION_MODEL": "mistralai/mistral-large-2512",
            "JUDGE_MODEL": "anthropic/claude-sonnet-5",
        }

        assert Settings.from_env(env).judge_model == "anthropic/claude-sonnet-5"
