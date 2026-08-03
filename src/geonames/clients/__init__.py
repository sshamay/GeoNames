"""Thin adapters for external systems (HTTP/API calls live here)."""

from geonames.clients.geonames_client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    GeoNamesClient,
    GeoNamesClientError,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "GeoNamesClient",
    "GeoNamesClientError",
]
