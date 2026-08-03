"""Thin adapters for external systems (HTTP/API calls live here)."""

from geonames.clients.geonames_client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    GeoNamesClient,
    GeoNamesClientError,
)
from geonames.clients.llm import (
    LlmClient,
    RealLlmClient,
    StubLlmClient,
    create_llm_client,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "GeoNamesClient",
    "GeoNamesClientError",
    "LlmClient",
    "RealLlmClient",
    "StubLlmClient",
    "create_llm_client",
]
