"""Data models for the "Ask about a location" assistant.

Request/response dataclasses exchanged between the question parser, the
GeoNames fetch layer and the LLM client. All values are plain, immutable
objects so the stub assistant is trivial to reason about and test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class Location:
    """A resolved location with a geographic centre point."""

    name: str
    lat: float
    lng: float


@dataclass(frozen=True)
class AssistantPlan:
    """The parsed intent of a user question: what to query and how."""

    question: str
    endpoints: Tuple[str, ...] = ()
    location: str = ""
    radius_km: float = 100.0
    time_window: str = "recent"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointResult:
    """One GeoNames endpoint call: its params and the raw response body."""

    endpoint: str
    params: Dict[str, Any]
    data: Dict[str, Any]
