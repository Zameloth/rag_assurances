"""SPEC §16.3 — every value the pipeline runs on comes from the environment."""

import pytest

from rag.config import ConfigurationError, Settings


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


class TestAbsentVariables:
    def test_the_model_ids_have_no_source_level_fallback(self) -> None:
        """A run must not resolve to an arm nobody chose — see the module docstring."""
        settings = Settings.from_env({})

        assert settings.generation_model == ""
        assert settings.condenser_model == ""
        assert settings.judge_model == ""

    def test_no_credential_is_demanded_of_a_stage_that_does_not_use_it(self) -> None:
        settings = Settings.from_env({})

        assert settings.openrouter_api_key == ""
        assert settings.langfuse_public_key == ""
        assert settings.langfuse_secret_key == ""

    def test_the_two_spec_constants_do_have_defaults(self) -> None:
        settings = Settings.from_env({})

        assert settings.langfuse_base_url == "https://cloud.langfuse.com"
        assert settings.qdrant_url == "http://localhost:6333"


class TestLangfuseTracing:
    """SPEC §11.2 — the free tier fails on debugging, so dev traces nothing."""

    def test_defaults_to_false_when_unset(self) -> None:
        assert Settings.from_env({}).langfuse_tracing is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "on"])
    def test_accepts_the_usual_truthy_spellings(self, value: str) -> None:
        assert Settings.from_env({"LANGFUSE_TRACING": value}).langfuse_tracing is True

    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off", ""])
    def test_accepts_the_usual_falsy_spellings(self, value: str) -> None:
        assert Settings.from_env({"LANGFUSE_TRACING": value}).langfuse_tracing is False

    def test_rejects_a_typo_rather_than_silently_reading_it_as_off(self) -> None:
        with pytest.raises(ConfigurationError, match="LANGFUSE_TRACING"):
            Settings.from_env({"LANGFUSE_TRACING": "ture"})


def test_langfuse_host_is_rejected_as_the_dead_sdk_v3_name() -> None:
    """SPEC §11.3 — v4 ignores the v3 name in silence, and EU cloud hides the mistake."""
    with pytest.raises(ConfigurationError, match="LANGFUSE_BASE_URL"):
        Settings.from_env({"LANGFUSE_HOST": "https://cloud.langfuse.com"})


class TestJudgePromptLanguages:
    """SPEC §12.10 — the FR/EN judge question is settled empirically, so both languages
    are config, and the starting point is the spec's own: managed Faithfulness v2 in
    English, the custom point-coverage evaluator in French."""

    def test_unset_resolves_to_the_spec_starting_configuration(self) -> None:
        settings = Settings.from_env({})

        assert settings.judge_faithfulness_language == "en"
        assert settings.judge_point_coverage_language == "fr"

    def test_each_metric_is_switched_independently(self) -> None:
        settings = Settings.from_env(
            {"JUDGE_FAITHFULNESS_LANGUAGE": "FR", "JUDGE_POINT_COVERAGE_LANGUAGE": " en "}
        )

        assert settings.judge_faithfulness_language == "fr"
        assert settings.judge_point_coverage_language == "en"

    def test_rejects_a_language_with_no_prompt_rather_than_falling_back(self) -> None:
        with pytest.raises(ConfigurationError, match="JUDGE_FAITHFULNESS_LANGUAGE"):
            Settings.from_env({"JUDGE_FAITHFULNESS_LANGUAGE": "de"})
