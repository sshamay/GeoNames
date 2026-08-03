"""Unit tests for the LLM client abstraction and the stub implementation."""

import pytest

from geonames.clients.llm import RealLlmClient, StubLlmClient, create_llm_client
from geonames.models.assistant import EndpointResult


def _result(endpoint, data):
    return EndpointResult(endpoint=endpoint, params={}, data=data)


QUAKES = {"earthquakes": [{"magnitude": 4.2}, {"magnitude": 3.1}, {"magnitude": 5.0}]}
WEATHER = {"weatherObservations": [{"ICAO": "KSMF"}, {"ICAO": "KSAC"}]}


@pytest.mark.unit
def test_stub_summarizes_earthquakes_with_strongest_magnitude():
    stub = StubLlmClient()
    summary = stub.summarize("question", [_result("earthquakes", QUAKES)], "Sacramento")
    assert "3 recent earthquakes" in summary
    assert "strongest magnitude 5.0" in summary
    assert "near Sacramento" in summary


@pytest.mark.unit
def test_stub_summarizes_weather_station_count():
    summary = StubLlmClient().summarize("question", [_result("weather", WEATHER)])
    assert "weather observations from 2 stations" in summary


@pytest.mark.unit
def test_stub_combines_multiple_endpoints():
    summary = StubLlmClient().summarize(
        "question", [_result("earthquakes", QUAKES), _result("weather", WEATHER)]
    )
    assert "3 recent earthquakes" in summary
    assert "weather observations from 2 stations" in summary


@pytest.mark.unit
def test_stub_handles_empty_earthquakes_and_weather():
    stub = StubLlmClient()
    assert "no recent earthquakes" in stub.summarize(
        "q", [_result("earthquakes", {})], "Sacramento"
    )
    assert "no weather observations" in stub.summarize(
        "q", [_result("weather", {"weatherObservations": []})]
    )


@pytest.mark.unit
def test_stub_returns_fallback_when_no_relevant_data():
    assert StubLlmClient().summarize("q", []) == "No GeoNames data was relevant to the question."


@pytest.mark.unit
def test_create_llm_client_defaults_to_stub():
    assert isinstance(create_llm_client(), StubLlmClient)


@pytest.mark.unit
def test_create_llm_client_uses_explicit_provider():
    assert isinstance(create_llm_client("stub"), StubLlmClient)
    assert isinstance(create_llm_client("real"), RealLlmClient)


@pytest.mark.unit
def test_create_llm_client_honours_environment_override(mocker):
    mocker.patch.dict("os.environ", {"GEONAMES_LLM_PROVIDER": "real"})
    assert isinstance(create_llm_client(), RealLlmClient)


@pytest.mark.unit
def test_create_llm_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_client("gpt-9000")


@pytest.mark.unit
def test_real_client_is_a_clear_placeholder():
    with pytest.raises(NotImplementedError, match="placeholder"):
        RealLlmClient().summarize("q", [_result("weather", WEATHER)])
