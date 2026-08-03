"""Unit tests for the Ask-about-a-location assistant orchestration.

Offline tests verify parse/fetch/summarize orchestration with fake fetchers,
without touching the network.
"""

import pytest

from geonames.clients.llm import StubLlmClient
from geonames.models.assistant import EndpointResult
from geonames.services.ask_location import AskLocationAssistant


@pytest.mark.unit
def test_assistant_plan_parses_without_network():
    assistant = AskLocationAssistant(llm_client=StubLlmClient(), fetchers={})
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
        llm_client=StubLlmClient(),
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
def test_assistant_forwards_results_to_llm(mocker):
    fake_llm = mocker.Mock()
    assistant = AskLocationAssistant(
        llm_client=fake_llm,
        fetchers={"weather": lambda params: {"weatherObservations": []}},
    )
    assistant.answer("Weather near Paris?")
    fake_llm.summarize.assert_called_once()
    results = fake_llm.summarize.call_args.args[1]
    assert isinstance(results[0], EndpointResult)
    assert results[0].endpoint == "weather"
    assert fake_llm.summarize.call_args.kwargs["location"] == "Paris"
