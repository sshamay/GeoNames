"""Dataclasses and models for data exchanged with the system under test."""

from geonames.models.assistant import (
    AssistantPlan,
    EndpointResult,
    Location,
)
from geonames.models.geonames import (
    Earthquake,
    EarthquakesResponse,
    FindNearbyResponse,
    StatusError,
    Toponym,
    WeatherObservation,
    WeatherResponse,
)

__all__ = [
    "AssistantPlan",
    "EndpointResult",
    "Location",
    "Earthquake",
    "EarthquakesResponse",
    "FindNearbyResponse",
    "StatusError",
    "Toponym",
    "WeatherObservation",
    "WeatherResponse",
]
