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
