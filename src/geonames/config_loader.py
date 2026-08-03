"""Load and validate YAML config into typed settings.

config/config.yaml is the single source of truth: env profiles (dev/test/staging)
are merged over defaults, and the rest of the code reads settings from here
instead of hardcoding values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigError(RuntimeError):
    """Raised when the config file is missing, invalid, or the env is unknown."""


@dataclass(frozen=True)
class Settings:
    """Typed view of the merged config for one environment."""

    app_name: str
    env: str
    geonames_username: Optional[str] = None
    geonames_base_url: str = "https://secure.geonames.org"
    geonames_timeout: float = 30.0
    assistant_max_rows: int = 5


def _default_config_path() -> Path:
    """Resolve config/config.yaml relative to the repo root.

    src/geonames/config_loader.py -> parents[2] is the project root.
    """
    return Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(env: str = "dev", config_path: Optional[Path] = None) -> Settings:
    """Load and merge the requested environment profile from config.yaml.

    Args:
        env: Profile name; must exist under ``envs`` in the YAML.
        config_path: Override the default config location (used in tests).

    Returns:
        A frozen Settings dataclass for the requested environment.

    Raises:
        ConfigError: If the file is missing, unparsable, or the env is unknown.
    """
    path = config_path or _default_config_path()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    envs = raw.get("envs", {})
    if env not in envs:
        raise ConfigError(f"Unknown environment '{env}'. Known: {sorted(envs)}")

    env_overrides = envs[env] or {}
    merged: Dict[str, Any] = {**raw.get("defaults", {}), **env_overrides}
    return Settings(
        app_name=merged.get("app_name", "geonames"),
        env=env,
        geonames_username=merged.get("geonames_username"),
        geonames_base_url=merged.get("geonames_base_url", "https://secure.geonames.org"),
        geonames_timeout=merged.get("geonames_timeout", 30.0),
        assistant_max_rows=merged.get("assistant_max_rows", 5),
    )
