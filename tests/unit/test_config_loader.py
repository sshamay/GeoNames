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
