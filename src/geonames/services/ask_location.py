"""The "Ask about a location" assistant service.

A thin workflow that turns one free-text question into an answer:

1. parse the question into an :class:`AssistantPlan` (endpoints + params)
2. resolve the location to coordinates and derive a bounding box
3. fetch raw data from the relevant GeoNames endpoints (via injected fetchers)
4. render a natural-language summary from the raw JSON

The GeoNames fetchers are injected as callables keyed by endpoint name, so the
service never imports the test-owned HTTP client; tests can wire real service
APIs or fakes in freely.

Summarization is a deterministic, offline function (``summarize``): it formats
the fetched JSON into text with no model call, so the full pipeline
(parse -> fetch -> summarize) is testable without a network.
"""

from __future__ import annotations

from dataclasses import asdict
from math import cos, radians
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from geonames.models.assistant import AssistantPlan, EndpointResult, Location
from geonames.services.intent import (
    LocationResolver,
    QuestionParser,
    UnknownIntentError,
    UnknownLocationError,
)
from geonames.tracing import TestTraceCollector, TraceCollector

Fetcher = Callable[[Dict[str, Any]], Dict[str, Any]]


class AskLocationAssistant:
    """Answers free-text location questions using GeoNames data.

    The assistant emits telemetry to its ``trace_collector`` as it runs (the
    parsed plan, the resolved location, and every GeoNames service call with
    its exact parameters), so tests and the golden-anchor evaluator can verify
    the execution path and the params each service received.
    """

    def __init__(
        self,
        fetchers: Mapping[str, Fetcher],
        parser: Optional[QuestionParser] = None,
        resolver: Optional[LocationResolver] = None,
        trace_collector: Optional[TraceCollector] = None,
    ) -> None:
        self._fetchers = dict(fetchers)
        self._parser = parser or QuestionParser()
        self._resolver = resolver or LocationResolver()
        self.trace_collector = trace_collector or TestTraceCollector()

    def plan(self, question: str) -> AssistantPlan:
        """Parse the question without fetching or summarizing anything."""
        return self._parser.parse(question)

    def answer(self, question: str) -> str:
        """Answer the question: parse, fetch, summarize (with telemetry).

        Non-query small talk (no weather/seismic intent, no location, or an
        unknown place) does not crash the assistant - it returns a graceful
        guidance reply instead of raising.
        """
        self.trace_collector.reset()
        try:
            plan = self._parser.parse(question)
            location = self._resolver.resolve(plan.location)
        except (UnknownIntentError, ValueError, UnknownLocationError):
            return _fallback_reply(question)
        self.trace_collector.on_plan(asdict(plan))
        self.trace_collector.on_location_resolved(asdict(location))

        results: list[EndpointResult] = []
        for endpoint in plan.endpoints:
            params = _endpoint_params(endpoint, location, plan)
            self.trace_collector.on_tool_called(endpoint, params)
            data = self._fetchers[endpoint](params)
            self.trace_collector.on_tool_output(endpoint, data)
            results.append(EndpointResult(endpoint=endpoint, params=params, data=data))
        return summarize(results, location=location.name)

    def process_user_query(self, user_input: str) -> str:
        """Golden-anchor contract alias for :meth:`answer`."""
        return self.answer(user_input)


def _fallback_reply(question: str) -> str:
    """Friendly guidance for input that is not a weather/seismic query."""
    return (
        "Hi, I'm a weather and earthquake assistant. Ask me about recent "
        "earthquakes or the weather near a city, for example 'Any recent "
        "earthquakes near London?'."
    )


def summarize(
    results: Sequence[EndpointResult],
    location: Optional[str] = None,
) -> str:
    """Render a deterministic summary of the fetched GeoNames data."""
    summaries: List[str] = []
    for result in results:
        if result.endpoint == "earthquakes":
            summaries.append(_summarize_earthquakes(result.data))
        elif result.endpoint == "weather":
            summaries.append(_summarize_weather(result.data))
    if not summaries:
        return "No GeoNames data was relevant to the question."
    where = f" near {location}" if location else ""
    return f"Found {_join(summaries)}{where}."


def _summarize_earthquakes(data: dict) -> str:
    quakes = data.get("earthquakes", []) or []
    if not quakes:
        return "no recent earthquakes"
    strongest = max(quake["magnitude"] for quake in quakes)
    return f"{len(quakes)} recent earthquakes (strongest magnitude {strongest:.1f})"


def _summarize_weather(data: dict) -> str:
    observations = data.get("weatherObservations", []) or []
    if not observations:
        return "no weather observations"
    return f"weather observations from {len(observations)} stations"


def _join(parts: List[str]) -> str:
    """Join summary fragments grammatically (empty-safe, no extra 'and' noise)."""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def bbox_from_center(location: Location, radius_km: float) -> Dict[str, float]:
    """A square bounding box of ``radius_km`` around a centre point.

    Latitude degrees are a constant distance; longitude degrees shrink with
    cos(lat). Clamped to avoid a degenerate box at extreme latitudes.
    """
    lat_delta = radius_km / 111.0
    lng_delta = radius_km / (111.0 * max(cos(radians(location.lat)), 0.05))
    return {
        "north": location.lat + lat_delta,
        "south": location.lat - lat_delta,
        "east": location.lng + lng_delta,
        "west": location.lng - lng_delta,
    }


def _endpoint_params(
    endpoint: str,
    location: Location,
    plan: AssistantPlan,
) -> Dict[str, Any]:
    """Build the API-level query params for one endpoint from the parsed plan.

    Params are flat (``north``/``south``/``east``/``west``, plus ``date`` for
    earthquakes) so the trace and the golden-anchor evaluator can assert them
    directly against what the service received.
    """
    params: Dict[str, Any] = bbox_from_center(location, plan.radius_km)
    if endpoint == "earthquakes" and plan.time_window != "recent":
        params["date"] = plan.time_window
    return params
