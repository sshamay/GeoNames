"""Unit tests for the host ``aqua_config.py`` discovery + precedence."""

import os

import pytest

from aqua.config import JudgeConfig, Thresholds
from aqua.plugin import (
    _load_host_config,
    resolve_hallucination_extractor,
    resolve_judge_config,
    resolve_thresholds,
)


@pytest.mark.unit
def test_load_host_config_returns_none_when_absent(tmp_path):
    assert _load_host_config(str(tmp_path / "missing.py")) is None


@pytest.mark.unit
def test_load_host_config_loads_module(tmp_path):
    (tmp_path / "aqua_config.py").write_text(
        "CASES_PATH = 'cases.json'\nREPORT_DIR = 'out'\n"
    )
    host = _load_host_config(str(tmp_path / "aqua_config.py"))
    assert host is not None
    assert host.CASES_PATH == "cases.json"
    assert host.REPORT_DIR == "out"


@pytest.mark.unit
def test_resolve_judge_config_prefers_host_config(tmp_path):
    (tmp_path / "aqua_config.py").write_text(
        "from aqua.config import JudgeConfig\n"
        "JUDGE = JudgeConfig(enabled=True, provider='ollama', model='m')\n"
    )
    host = _load_host_config(str(tmp_path / "aqua_config.py"))
    judge = resolve_judge_config(host)
    assert judge.enabled is True
    assert judge.provider == "ollama"
    assert judge.model == "m"


@pytest.mark.unit
def test_resolve_judge_config_falls_back_to_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AQUA_JUDGE_PROVIDER", "aihorde")
    monkeypatch.setenv("AQUA_JUDGE_MODEL", "google/gemma-4-31b")
    judge = resolve_judge_config(None)
    assert judge.provider == "aihorde"
    assert judge.model == "google/gemma-4-31b"


@pytest.mark.unit
def test_resolve_hallucination_extractor_from_host(tmp_path):
    (tmp_path / "aqua_config.py").write_text(
        "def hallucination_extractor(ai_output, tool_outputs):\n"
        "    return ['count 5 != 3']\n"
    )
    host = _load_host_config(str(tmp_path / "aqua_config.py"))
    extractor = resolve_hallucination_extractor(host)
    assert extractor("reply", {}) == ["count 5 != 3"]


@pytest.mark.unit
def test_resolve_hallucination_extractor_none_when_absent(tmp_path):
    (tmp_path / "aqua_config.py").write_text("CASES_PATH = 'cases.json'\n")
    host = _load_host_config(str(tmp_path / "aqua_config.py"))
    assert resolve_hallucination_extractor(host) is None


@pytest.mark.unit
def test_resolve_thresholds_prefers_host_config(tmp_path):
    (tmp_path / "aqua_config.py").write_text(
        "from aqua.config import Thresholds\n"
        "THRESHOLDS = Thresholds(expected_outcome_semantic=0.5, default_case=0.95)\n"
    )
    host = _load_host_config(str(tmp_path / "aqua_config.py"))
    thresholds = resolve_thresholds(host)
    assert thresholds.expected_outcome_semantic == 0.5
    assert thresholds.llm_judge_pass == 0.3
    assert thresholds.default_case == 0.95


@pytest.mark.unit
def test_resolve_thresholds_defaults_when_absent(tmp_path):
    (tmp_path / "aqua_config.py").write_text("CASES_PATH = 'cases.json'\n")
    host = _load_host_config(str(tmp_path / "aqua_config.py"))
    thresholds = resolve_thresholds(host)
    assert thresholds == Thresholds()
