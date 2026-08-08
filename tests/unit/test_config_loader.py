"""Unit tests for the config loader (the package's entry point to config)."""

from pathlib import Path

import pytest

from geonames.config_loader import ConfigError, load_config


@pytest.mark.unit
def test_load_config_returns_test_profile(settings):
    """The session fixture yields the merged ``test`` profile from config.yaml."""
    assert settings.env == "test"
    assert settings.app_name == "geonames"


@pytest.mark.unit
@pytest.mark.parametrize("env", ["dev", "test", "staging"])
def test_load_config_supports_all_defined_envs(env):
    """Every profile declared in config.yaml loads without errors."""
    settings = load_config(env=env)
    assert settings.env == env


@pytest.mark.unit
def test_load_config_raises_on_unknown_env():
    """Unknown profiles fail fast instead of silently merging an empty dict."""
    with pytest.raises(ConfigError, match="Unknown environment 'prod'"):
        load_config(env="prod")


@pytest.mark.unit
def test_load_config_raises_when_file_missing():
    """A missing config file is a clear, specific error - not a bare exception."""
    missing_path = Path("/nonexistent/config.yaml")
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(config_path=missing_path)


@pytest.mark.unit
def test_judge_config_loaded_from_yaml(tmp_path):
    """YAML judge keys map onto Settings (deterministic, not local-config dependent)."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "defaults:\n"
        "  app_name: geonames\n"
        "  judge_enabled: true\n"
        "  judge_base_url: http://localhost:11434/v1\n"
        "  judge_model: llama3.2:3b\n"
        "  judge_api_key: ollama\n"
        "  judge_timeout: 60\n"
        "  judge_rubric: groundedness_and_completeness\n"
        "  judge_debug: true\n"
        "envs:\n  test:\n"
    )
    settings = load_config(env="test", config_path=config_file)
    assert settings.judge_enabled is True
    assert settings.judge_base_url == "http://localhost:11434/v1"
    assert settings.judge_model == "llama3.2:3b"
    assert settings.judge_api_key == "ollama"
    assert settings.judge_timeout == 60
    assert settings.judge_rubric == "groundedness_and_completeness"
    assert settings.judge_debug is True


@pytest.mark.unit
def test_judge_fallback_defaults_without_keys(tmp_path):
    """Missing judge keys fall back to safe defaults (offline) instead of raising."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("defaults:\n  app_name: geonames\nenvs:\n  test:\n")
    settings = load_config(env="test", config_path=config_file)
    assert settings.judge_enabled is False
    assert settings.judge_base_url is None
    assert settings.judge_model is None
    assert settings.judge_api_key is None
    assert settings.judge_timeout == 60.0
    assert settings.judge_rubric == "groundedness_and_completeness"
    assert settings.judge_debug is False
