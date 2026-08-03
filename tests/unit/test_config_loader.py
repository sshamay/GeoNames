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
def test_judge_config_loaded_from_yaml(settings):
    """The local config.yaml maps judge keys onto Settings."""
    assert settings.judge_enabled is True
    assert settings.judge_base_url == "https://oai.aihorde.net/v1"
    assert settings.judge_model == "google/gemma-4-31b"
    assert settings.judge_api_key == "0000000000"
    assert settings.judge_timeout == 120


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
