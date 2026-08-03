"""Unit tests for the AQuA evaluator's per-case KPIs (parameter-aware intent)."""

import pytest

from test_utils.aqua_evaluation import AQuAEvaluators, required_tool_executed

pytestmark = pytest.mark.unit


def _trace(executed=None, calls=None):
    return {
        "executed_tools": executed or [],
        "executed_tool_calls": calls or [],
        "tool_outputs": {},
    }


@pytest.mark.unit
def test_intent_true_when_name_and_params_match():
    case = {
        "required_tools": [
            {"name": "earthquakes", "parameters": {"north": 1.0, "south": 0.0}},
        ]
    }
    trace = _trace(
        executed=["earthquakes"],
        calls=[{"name": "earthquakes", "parameters": {"north": 1.0, "south": 0.0, "east": 2.0}}],
    )
    assert AQuAEvaluators.compute_metrics(case, "out", trace)["intent_accurate"] is True


@pytest.mark.unit
def test_intent_false_when_params_wrong():
    case = {"required_tools": [{"name": "earthquakes", "parameters": {"north": 1.0}}]}
    trace = _trace(
        executed=["earthquakes"],
        calls=[{"name": "earthquakes", "parameters": {"north": 9.9}}],
    )
    assert AQuAEvaluators.compute_metrics(case, "out", trace)["intent_accurate"] is False


@pytest.mark.unit
def test_intent_false_when_extra_tool_called():
    case = {"required_tools": ["earthquakes"]}
    trace = _trace(executed=["earthquakes", "weather"], calls=[])
    assert AQuAEvaluators.compute_metrics(case, "out", trace)["intent_accurate"] is False


@pytest.mark.unit
def test_intent_none_when_no_tools_declared():
    case = {"required_tools": []}
    assert AQuAEvaluators.compute_metrics(case, "out", _trace())["intent_accurate"] is None


@pytest.mark.unit
def test_intent_falls_back_to_name_when_no_call_trace():
    case = {"required_tools": [{"name": "weather", "parameters": {"north": 1.0}}]}
    trace = _trace(executed=["weather"], calls=[])
    assert AQuAEvaluators.compute_metrics(case, "out", trace)["intent_accurate"] is True


@pytest.mark.unit
def test_required_tool_executed_name_only():
    assert required_tool_executed("earthquakes", ["earthquakes"], []) is True
    assert required_tool_executed("earthquakes", ["weather"], []) is False
    assert required_tool_executed("earthquakes", ["earthquakes", "weather"], []) is True


def _fake_judge(score, reason="judge decided"):
    def _j(ai_output, expected_outcome, retrieved_context):
        _j.calls += 1
        _j.last_context = retrieved_context
        return {"score": score, "reason": reason}

    _j.calls = 0
    _j.last_context = None
    return _j


@pytest.mark.unit
def test_expected_outcome_no_judge_fails_below_threshold(monkeypatch):
    monkeypatch.setattr(AQuAEvaluators, "llm_judge", None)
    result = AQuAEvaluators.evaluate_expected_outcome(
        "banana spaceship quantum jelly", "the stock market rose today"
    )
    assert result["status"] == "FAILED"
    assert result["check_name"] == "expected_outcome"


@pytest.mark.unit
def test_expected_outcome_escalates_to_judge_when_below_threshold(monkeypatch):
    judge = _fake_judge(0.9)
    monkeypatch.setattr(AQuAEvaluators, "llm_judge", judge)
    result = AQuAEvaluators.evaluate_expected_outcome(
        "banana spaceship quantum jelly", "the stock market rose today", ["ctx-a", "ctx-b"]
    )
    assert result["check_name"] == "llm_judge"
    assert result["status"] == "PASSED"
    assert result["score"] == 0.9
    assert judge.calls == 1
    assert judge.last_context == ["ctx-a", "ctx-b"]


@pytest.mark.unit
def test_expected_outcome_judge_fail(monkeypatch):
    judge = _fake_judge(0.2)
    monkeypatch.setattr(AQuAEvaluators, "llm_judge", judge)
    result = AQuAEvaluators.evaluate_expected_outcome(
        "banana spaceship quantum jelly", "the stock market rose today"
    )
    assert result["check_name"] == "llm_judge"
    assert result["status"] == "FAILED"
    assert result["score"] == 0.2


@pytest.mark.unit
def test_expected_outcome_judge_not_called_on_deterministic_pass(monkeypatch):
    judge = _fake_judge(0.9)
    monkeypatch.setattr(AQuAEvaluators, "llm_judge", judge)
    result = AQuAEvaluators.evaluate_expected_outcome(
        "the stock market rose today", "the stock market rose today"
    )
    assert result["status"] == "PASSED"
    assert result["check_name"] == "expected_outcome"
    assert judge.calls == 0


@pytest.mark.unit
def test_run_case_escalates_to_judge_and_scores_confidence(monkeypatch):
    judge = _fake_judge(0.9)
    monkeypatch.setattr(AQuAEvaluators, "llm_judge", judge)
    case = {
        "expected_outcome": "the stock market rose today",
        "threshold": 0.9,
    }
    result = AQuAEvaluators.run_case(
        case, "banana spaceship quantum jelly", {"retrieved_context": ["doc-1"]}
    )
    judge_check = next(c for c in result["details"] if c["check_name"] == "llm_judge")
    assert judge_check["status"] == "PASSED"
    assert result["aggregate_score"] == 0.9
    assert result["action"] == "RELEASE"
