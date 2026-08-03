"""Tests for the assistant's telemetry (trace collector).

The trace follows the SmartSpend golden-anchor contract: ``executed_tools``
holds the execution path (service call order) and ``executed_tool_calls`` the
exact parameters each service received.
"""

import pytest

from geonames.services.ask_location import AskLocationAssistant
from geonames.tracing import TestTraceCollector


@pytest.mark.unit
def test_trace_collector_records_path_and_params():
    collector = TestTraceCollector()
    collector.on_tool_called("earthquakes", {"north": 1.0})
    collector.on_tool_called("weather", {"north": 1.0, "date": "2023-02-06"})

    logs = collector.get_trace_logs()
    assert logs["executed_tools"] == ["earthquakes", "weather"]
    assert logs["executed_tool_calls"] == [
        {"name": "earthquakes", "parameters": {"north": 1.0}},
        {"name": "weather", "parameters": {"north": 1.0, "date": "2023-02-06"}},
    ]


@pytest.mark.unit
def test_trace_collector_reset_clears_events():
    collector = TestTraceCollector()
    collector.on_tool_called("earthquakes", {})
    collector.on_plan({"question": "q"})
    collector.reset()
    logs = collector.get_trace_logs()
    assert logs["executed_tools"] == []
    assert logs["executed_tool_calls"] == []
    assert logs["plan"] == {}


@pytest.mark.unit
def test_assistant_emits_plan_location_and_service_calls_in_order():
    calls = []

    def _fetcher(endpoint, payload):
        def _f(params):
            calls.append((endpoint, params))
            return payload

        return _f

    assistant = AskLocationAssistant(
        fetchers={
            "earthquakes": _fetcher("earthquakes", {"earthquakes": []}),
            "weather": _fetcher("weather", {"weatherObservations": []}),
        },
    )

    assistant.answer("Any recent earthquakes or bad weather near Sacramento?")

    logs = assistant.trace_collector.get_trace_logs()
    assert logs["executed_tools"] == ["earthquakes", "weather"]
    assert logs["plan"]["location"] == "Sacramento"
    assert logs["location"]["name"] == "Sacramento"

    quake_params = logs["executed_tool_calls"][0]["parameters"]
    weather_params = logs["executed_tool_calls"][1]["parameters"]
    assert {"north", "south", "east", "west"} <= set(quake_params)
    assert {"north", "south", "east", "west"} <= set(weather_params)
    assert calls == [(endpoint, params) for endpoint, params in
                     [(c["name"], c["parameters"]) for c in logs["executed_tool_calls"]]]


@pytest.mark.unit
def test_assistant_trace_resets_between_answers():
    assistant = AskLocationAssistant(
        fetchers={"weather": lambda params: {"weatherObservations": []}},
    )
    assistant.answer("Weather near Paris?")
    assistant.answer("Weather near Tokyo?")

    logs = assistant.trace_collector.get_trace_logs()
    assert logs["executed_tools"] == ["weather"]
    assert logs["plan"]["location"] == "Tokyo"


@pytest.mark.unit
def test_trace_logs_are_golden_anchor_compatible():
    """The generic evaluator keys are present and non-empty after a run."""
    assistant = AskLocationAssistant(
        fetchers={"weather": lambda params: {"weatherObservations": []}},
    )
    assistant.answer("Weather near London?")
    logs = assistant.trace_collector.get_trace_logs()
    assert set(("executed_tools", "executed_tool_calls", "retrieved_context")) <= set(logs)
    assert "weather" in logs["executed_tools"]
