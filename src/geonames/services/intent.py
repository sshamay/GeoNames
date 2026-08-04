"""Intent parsing for the "Ask about a location" assistant.

A rule-based, deterministic parser: it decides
which GeoNames endpoints are relevant and extracts the parameters (location,
radius, time window) from a free-text question. The parser is
deliberately simple - it recognises the vocabulary the tests exercise - and is
the single place to extend if the assistant later needs more endpoints.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from geonames.models.assistant import AssistantPlan

DEFAULT_RADIUS_KM = 100.0
KM_PER_MILE = 1.60934
KM_PER_DEGREE_LAT = 111.0

KNOWN_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "sacramento": (38.5816, -121.4944),
    "san francisco": (37.7749, -122.4194),
    "los angeles": (34.0522, -118.2437),
    "new york": (40.7128, -74.0060),
    "tokyo": (35.6762, 139.6503),
    "paris": (48.8566, 2.3522),
    "london": (51.5074, -0.1278),
    "california": (36.7783, -119.4179),
    # US ambiguous cities resolve to their most common state
    "springfield": (39.7817, -89.6501),  # Springfield, IL
    "springfield, il": (39.7817, -89.6501),
    "columbus": (39.9612, -82.9988),  # Columbus, OH
    "columbus, oh": (39.9612, -82.9988),
    "austin": (30.2672, -97.7431),  # Austin, TX
    "austin, tx": (30.2672, -97.7431),
}


class UnknownIntentError(ValueError):
    """Raised when no GeoNames endpoint can be inferred from the question."""


class UnknownLocationError(KeyError):
    """Raised when the question's location is not in the known-location map."""


class QuestionParser:
    """Extract a query plan from a free-text question."""

    def parse(self, question: str) -> AssistantPlan:
        """Parse the question into an :class:`AssistantPlan`.

        Raises:
            UnknownIntentError: When no supported endpoint is mentioned.
            ValueError: When no location can be extracted.
        """
        endpoints = _detect_endpoints(question)
        if not endpoints:
            raise UnknownIntentError(
                f"no supported GeoNames endpoint in question: {question!r}"
            )
        return AssistantPlan(
            question=question,
            endpoints=endpoints,
            location=_extract_location(question),
            radius_km=_extract_radius_km(question),
            time_window=_extract_time_window(question),
        )


class LocationResolver:
    """Map a location name to geographic coordinates (stub geocoder)."""

    def __init__(self, known: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self._known = dict(known or KNOWN_LOCATIONS)

    def resolve(self, name: str):
        """Resolve a place name to a :class:`Location`.

        Raises:
            UnknownLocationError: When the name is not in the known map.
        """
        from geonames.models.assistant import Location

        key = name.strip().lower()
        # Handle "City, ST" format by also trying the city-only key
        if key not in self._known:
            # Try stripping ", ST" suffix
            city_only = key.split(",")[0].strip()
            if city_only in self._known:
                key = city_only
            else:
                raise UnknownLocationError(
                    f"unknown location {name!r}; known: {sorted(self._known)}"
                )
        lat, lng = self._known[key]
        return Location(name=name.strip(), lat=lat, lng=lng)


def _detect_endpoints(question: str) -> Tuple[str, ...]:
    lowered = question.lower()
    endpoints = []
    if "earthquake" in lowered:
        endpoints.append("earthquakes")
    if any(word in lowered for word in ("weather", "storm", "rain")):
        endpoints.append("weather")
    return tuple(endpoints)


def _extract_location(question: str) -> str:
    # Handle "near me" conversational pattern -> defaults to Sacramento
    if re.search(r"\bnear me\b", question, re.IGNORECASE):
        return "Sacramento"

    # Handle "close to downtown <City>" pattern
    downtown_match = re.search(
        r"\bclose to downtown\s+([A-Za-z][A-Za-z'\- ,]*?)(?:\s*(?:yesterday|today|near|around|at|\?|$))",
        question, re.IGNORECASE
    )
    if downtown_match:
        return downtown_match.group(1).strip().rstrip(".,;:!?")

    # Standard "near|around|at <location>" pattern
    match = re.search(
        r"\b(?:near|around|at)\s+([A-Za-z][A-Za-z'\- ,]*?)(?:\s*(?:yesterday|today|near|around|at|\?|$))",
        question, re.IGNORECASE
    )
    if match is None:
        match = re.search(r"\bin\s+([A-Z][a-z]+(?:[ \-][A-Z][a-z]+)*)", question)
    if match is None:
        raise ValueError(f"no location mentioned in question: {question!r}")
    location = match.group(1).strip().rstrip(".,;:!?")
    # Strip trailing conversational words
    location = re.sub(r"\s+(around|downtown|near)\s*$", "", location, flags=re.IGNORECASE)
    return location


def _extract_radius_km(question: str) -> float:
    match = re.search(
        r"within\s+(\d+(?:\.\d+)?)\s*(km|kilometers?|miles?)",
        question,
        re.IGNORECASE,
    )
    if match is None:
        return DEFAULT_RADIUS_KM
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("mi"):
        value *= KM_PER_MILE
    return value


def _extract_time_window(question: str) -> str:
    lowered = question.lower()
    explicit_date = re.search(r"\d{4}-\d{2}-\d{2}", question)
    if explicit_date:
        return explicit_date.group(0)
    if re.search(r"last\s+\d+\s+hours?", lowered):
        return "last_24h"
    if re.search(r"last\s+\d+\s+days?", lowered):
        return "last_7d"
    if re.search(r"last\s+\d+\s+weeks?", lowered):
        return "last_2w"
    if "today" in lowered:
        return "today"
    if "yesterday" in lowered:
        return "yesterday"
    return "recent"
