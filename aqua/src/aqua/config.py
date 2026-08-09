"""Framework configuration for the AQuA evaluation toolkit.

Portable by design: the framework never reads the host project's config file.
Judge settings come from ``AQUA_JUDGE_*`` environment variables (or a
programmatic :class:`JudgeConfig`), so the framework runs identically in any
project. A host project may override the pytest ``aqua_judge_config`` fixture
to source the same values from its own config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgeConfig:
    """Judge settings independent of any host project's config format."""

    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 60.0
    rubric: str = "groundedness_and_completeness"
    debug: bool = False
    system_prompt: Optional[str] = None
    max_tokens: int = 256  # cap on judge model output tokens
    max_attempts: int = 3  # retries on transport errors / unparseable verdicts


@dataclass
class Thresholds:
    """Evaluation gate thresholds (overridable from the host's aqua_config.py).

    Mirrors the framework defaults in ``aqua.evaluation``; a host project sets
    ``THRESHOLDS`` in its ``aqua_config.py`` to tune them in one place.
    """

    # Similarity required for the cheap semantic layer (P5) to pass a case
    # without escalation to the LLM judge.
    expected_outcome_semantic: float = 0.6
    # Minimum judge score (0..1) for the P6 LLM-as-a-Judge check to pass.
    llm_judge_pass: float = 0.3
    # Aggregate confidence below which a case escalates to HITL / fails
    # (used when the case JSON does not declare its own `threshold`).
    default_case: float = 0.9


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def judge_config_from_env() -> JudgeConfig:
    """Build a JudgeConfig from AQUA_JUDGE_* environment variables."""
    return JudgeConfig(
        enabled=_env_bool("AQUA_JUDGE_ENABLED"),
        provider=os.environ.get("AQUA_JUDGE_PROVIDER", "openai_compatible"),
        base_url=os.environ.get("AQUA_JUDGE_BASE_URL") or None,
        model=os.environ.get("AQUA_JUDGE_MODEL") or None,
        api_key=os.environ.get("AQUA_JUDGE_API_KEY") or None,
        timeout=float(os.environ.get("AQUA_JUDGE_TIMEOUT", 60)),
        rubric=os.environ.get("AQUA_JUDGE_RUBRIC", "groundedness_and_completeness"),
        debug=_env_bool("AQUA_JUDGE_DEBUG"),
        system_prompt=os.environ.get("AQUA_JUDGE_SYSTEM_PROMPT") or None,
        max_tokens=int(os.environ.get("AQUA_JUDGE_MAX_TOKENS", 256)),
        max_attempts=int(os.environ.get("AQUA_JUDGE_MAX_ATTEMPTS", 3)),
    )
