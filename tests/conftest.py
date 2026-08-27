"""Shared fixtures for the GeoNames test suite.

PROJECT-SPECIFIC ADAPTER. The generic golden-anchor harness (parametrization,
run ledger, reporting, dashboard, evaluators) lives in the ``aqua`` framework
and is auto-registered by its pytest plugin (entry point ``aqua.plugin``).

All AQuA configuration for this project - paths, thresholds, judge, and the
hallucination definition - lives in ONE place: the project-root ``aqua_config.py``
(the framework package itself ships only the generic template
``aqua/src/aqua/aqua_config.example.py``).

This conftest only wires the GeoNames side:

- the test environment settings (loaded through the exact same code path as
  production: geonames.config_loader.load_config),
- the shared GeoNames client + service APIs,
- the assistant (system under test).
"""

from __future__ import annotations

import pytest

from geonames.clients import GeoNamesClient
from geonames.config_loader import Settings, load_config
from geonames.services import EarthquakesAPI, WeatherAPI
from test_utils.find_nearby import FindNearbyAPI


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings for the ``test`` environment, merged from config.yaml."""
    return load_config(env="test")


@pytest.fixture(scope="session")
def client(settings: Settings) -> GeoNamesClient:
    """Single shared GeoNamesClient built from the test settings."""
    return GeoNamesClient(
        username=settings.geonames_username,
        base_url=settings.geonames_base_url,
        timeout=settings.geonames_timeout,
    )


@pytest.fixture(scope="session")
def earthquakes_api(client: GeoNamesClient) -> EarthquakesAPI:
    """Earthquakes service API bound to the shared client."""
    return EarthquakesAPI(client)


@pytest.fixture(scope="session")
def find_nearby_api(client: GeoNamesClient) -> FindNearbyAPI:
    """FindNearby service API bound to the shared client."""
    return FindNearbyAPI(client)


@pytest.fixture(scope="session")
def weather_api(client: GeoNamesClient) -> WeatherAPI:
    """Weather service API bound to the shared client."""
    return WeatherAPI(client)


# =============================================================================
# ASSISTANT (system under test)
# =============================================================================


@pytest.fixture(scope="session")
def assistant_class():
    """PROJECT-SPECIFIC: the GeoNames AskLocationAssistant class."""
    from geonames.services.ask_location import AskLocationAssistant

    return AskLocationAssistant


@pytest.fixture
def ai_assistant(assistant_class, settings):
    """PROJECT-SPECIFIC: assistant with production wiring decided by Settings.

    Uses build_assistant so the SUT owns its own composition (client, service
    fetchers, max_rows) instead of the test assembling it. Exposes
    ``process_user_query(input)`` and ``trace_collector.get_trace_logs()`` per
    the AQuA golden-anchor SUT contract.
    """
    from geonames.factory import build_assistant

    assistant = build_assistant(settings)
    assert isinstance(assistant, assistant_class)
    return assistant
