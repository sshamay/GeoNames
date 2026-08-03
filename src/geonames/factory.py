"""Production assembly for the "Ask about a location" assistant.

The SUT owns its own composition: build_assistant() wires the real GeoNames
client, the service fetchers (max_rows from Settings) and the LLM client
(resolved via create_llm_client from Settings.llm_provider), so tests exercise
the assistant as-is instead of re-assembling it.
"""

from __future__ import annotations

from typing import Optional

from geonames.clients.geonames_client import GeoNamesClient
from geonames.clients.llm import LlmClient, create_llm_client
from geonames.config_loader import Settings
from geonames.services.ask_location import AskLocationAssistant
from geonames.services.earthquakes import EarthquakesAPI
from geonames.services.weather import WeatherAPI


def build_geonames_client(
    settings: Settings,
    session: Optional[object] = None,
) -> GeoNamesClient:
    """A GeoNamesClient configured from Settings (optional transport session)."""
    return GeoNamesClient(
        username=settings.geonames_username,
        base_url=settings.geonames_base_url,
        timeout=settings.geonames_timeout,
        session=session,
    )


def build_assistant(
    settings: Settings,
    client: Optional[GeoNamesClient] = None,
    llm_client: Optional[LlmClient] = None,
) -> AskLocationAssistant:
    """Assemble the assistant with production wiring decided by Settings.

    Args:
        settings: Merged config (username, base url, timeout, llm provider,
            assistant_max_rows).
        client: Optional prebuilt GeoNamesClient (dependency injection).
        llm_client: Optional prebuilt LlmClient (dependency injection).

    Returns:
        A fully wired AskLocationAssistant.
    """
    client = client or build_geonames_client(settings)
    earthquakes_api = EarthquakesAPI(client)
    weather_api = WeatherAPI(client)
    max_rows = settings.assistant_max_rows

    def earthquakes_fetcher(params):
        return earthquakes_api.get_earthquakes(**params, max_rows=max_rows)

    def weather_fetcher(params):
        return weather_api.get_weather(**params, max_rows=max_rows)

    return AskLocationAssistant(
        llm_client=llm_client or create_llm_client(settings.llm_provider),
        fetchers={"earthquakes": earthquakes_fetcher, "weather": weather_fetcher},
    )
