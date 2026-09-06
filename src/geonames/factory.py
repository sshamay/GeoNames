"""Production assembly for the GeoNames agent.

The SUT owns its own composition: build_assistant() wires the real GeoNames
client, the service APIs (search/earthquakes/weather), the LangChain tools and
LLM, so tests exercise the agent as-is instead of re-assembling it.
"""

from __future__ import annotations

from typing import Optional

from langchain_ollama import ChatOllama

from geonames.clients.geonames_client import GeoNamesClient
from geonames.config_loader import AgentConfig, Settings
from geonames.services.agent import GeoNamesAgent
from geonames.services.earthquakes import EarthquakesAPI
from geonames.services.search import SearchAPI
from geonames.services.tools import GeoNamesTools
from geonames.services.weather import WeatherAPI
from geonames.telemetry import get_tracer, init_telemetry
from geonames.tracing import TestTraceCollector, TraceCollector


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


def build_llm(agent: AgentConfig) -> ChatOllama:
    """A ChatOllama model configured from the agent settings."""
    return ChatOllama(
        model=agent.model,
        base_url=agent.base_url,
        temperature=agent.temperature,
    )


def build_assistant(
    settings: Settings,
    client: Optional[GeoNamesClient] = None,
    trace_collector: Optional[TraceCollector] = None,
    llm=None,
) -> GeoNamesAgent:
    """Assemble the GeoNames agent with production wiring decided by Settings.

    Args:
        settings: Merged config (username, base url, timeout, agent settings).
        client: Optional prebuilt GeoNamesClient (dependency injection).
        trace_collector: Optional trace collector (OTel-enabled or in-memory).
        llm: Optional LLM (defaults to ChatOllama from ``settings.agent``).

    Returns:
        A fully wired GeoNamesAgent.
    """
    client = client or build_geonames_client(settings)
    max_rows = settings.agent.max_rows

    search_api = SearchAPI(client)
    earthquakes_api = EarthquakesAPI(client)
    weather_api = WeatherAPI(client)
    tools = GeoNamesTools(
        search_api=search_api,
        earthquakes_api=earthquakes_api,
        weather_api=weather_api,
        max_rows=max_rows,
        default_location=settings.default_location,
    )

    llm = llm or build_llm(settings.agent)

    if trace_collector is None:
        # Only engage OTel when explicitly requested; default is a pure
        # in-memory collector so unit/AQuA runs stay offline.
        collector: TraceCollector = TestTraceCollector()
    else:
        collector = trace_collector

    return GeoNamesAgent(
        tools=tools,
        llm=llm,
        trace_collector=collector,
        max_iterations=settings.agent.max_iterations,
    )
