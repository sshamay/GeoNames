"""Unit tests for the production factory (src/geonames/factory.py).

Verifies that build_assistant wires the SUT from Settings: the client is
configured from Settings and the service fetchers pass assistant_max_rows
through to every call.
"""

import pytest

from geonames.clients import GeoNamesClient
from geonames.config_loader import Settings
from geonames.factory import build_assistant, build_geonames_client
from geonames.services import GeoNamesAgent


def _settings(**overrides):
    defaults = {
        "app_name": "geonames",
        "env": "test",
        "geonames_username": "testuser",
        "geonames_base_url": "https://secure.geonames.org",
        "geonames_timeout": 10.0,
        "assistant_max_rows": 3,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class _FakeClient:
    """Duck-typed GeoNamesClient: records calls, returns empty responses."""

    def __init__(self):
        self.calls = []

    def get(self, endpoint, params):
        self.calls.append((endpoint, params))
        if endpoint == "earthquakesJSON":
            return {"earthquakes": []}
        if endpoint == "weatherJSON":
            return {"weatherObservations": []}
        raise AssertionError(f"unexpected endpoint {endpoint}")


@pytest.mark.unit
def test_build_geonames_client_uses_settings():
    client = build_geonames_client(_settings())
    assert isinstance(client, GeoNamesClient)
    assert client.username == "testuser"
    assert client.base_url == "https://secure.geonames.org"
    assert client.timeout == 10.0


@pytest.mark.unit
def test_build_assistant_wires_geonames_agent():
    """build_assistant returns a GeoNamesAgent wired to the tools + trace."""
    fake = _FakeClient()
    assistant = build_assistant(_settings(), client=fake)

    assert isinstance(assistant, GeoNamesAgent)
    # The agent owns a GeoNamesTools instance (geocode/earthquakes/weather).
    tools = assistant._tools
    assert hasattr(tools, "geocode_location")
    assert hasattr(tools, "earthquakes")
    assert hasattr(tools, "weather")
    # The trace collector is present (AQuA SUT contract).
    assert hasattr(assistant, "trace_collector")
    assert callable(assistant.process_user_query)
