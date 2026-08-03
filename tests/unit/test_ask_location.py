"""Unit tests for the Ask-about-a-location assistant orchestration.

Offline tests verify parse/fetch/summarize orchestration with fake fetchers,
without touching the network.
"""

import pytest

from geonames.models.assistant import EndpointResult
from geonames.services.ask_location import AskLocationAssistant, summarize


def _result(endpoint, data):
    return EndpointResult(endpoint=endpoint, params={}, data=data)


@pytest.mark.unit
def test_assistant_plan_parses_without_network():
    assistant = AskLocationAssistant(fetchers={})
    plan = assistant.plan("Earthquakes within 50 km near Tokyo?")
    assert plan.endpoints == ("earthquakes",)
    assert plan.location == "Tokyo"
    assert plan.radius_km == 50.0


@pytest.mark.unit
def test_assistant_fetches_each_planned_endpoint_and_summarizes():
    calls = {}

    def _fetcher(endpoint, payload):
        def _f(params):
            calls[endpoint] = params
            return payload

        return _f

    assistant = AskLocationAssistant(
        fetchers={
            "earthquakes": _fetcher("earthquakes", {"earthquakes": [{"magnitude": 4.2}]}),
            "weather": _fetcher("weather", {"weatherObservations": []}),
        },
    )

    summary = assistant.answer("Any recent earthquakes or bad weather near Sacramento?")

    assert set(calls) == {"earthquakes", "weather"}
    assert "near Sacramento" in summary
    assert "1 recent earthquakes" in summary
    for params in calls.values():
        assert {"north", "south", "east", "west"} <= set(params)


@pytest.mark.unit
def test_summarize_renders_earthquakes_and_weather():
    quakes = {"earthquakes": [{"magnitude": 4.2}, {"magnitude": 3.1}, {"magnitude": 5.0}]}
    weather = {"weatherObservations": [{"ICAO": "KSMF"}, {"ICAO": "KSAC"}]}
    text = summarize(
        [_result("earthquakes", quakes), _result("weather", weather)],
        location="Sacramento",
    )
    assert "3 recent earthquakes" in text
    assert "strongest magnitude 5.0" in text
    assert "weather observations from 2 stations" in text
    assert "near Sacramento" in text


@pytest.mark.unit
def test_summarize_handles_empty_data():
    assert "no recent earthquakes" in summarize(
        [_result("earthquakes", {})], location="Sacramento"
    )
    assert "no weather observations" in summarize(
        [_result("weather", {"weatherObservations": []})]
    )


@pytest.mark.unit
def test_summarize_falls_back_when_no_relevant_data():
    assert summarize([]) == "No GeoNames data was relevant to the question."
