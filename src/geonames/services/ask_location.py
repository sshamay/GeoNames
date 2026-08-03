"""The "Ask about a location" assistant service.

A thin workflow that orchestrates the pieces into one user-facing answer:

1. parse the question into an :class:`AssistantPlan` (endpoints + params)
2. resolve the location to coordinates and derive a bounding box
3. fetch raw data from the relevant GeoNames endpoints (via injected fetchers)
4. hand the question + raw JSON to the :class:`LlmClient` for a summary

The GeoNames fetchers are injected as callables keyed by endpoint name, so the
service never imports the test-owned HTTP client; tests can wire real service
APIs or fakes in freely.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from math import cos, radians
from typing import Any, Callable, Dict, Mapping, Optional

from geonames.clients.llm import LlmClient
from geonames.models.assistant import AssistantPlan, EndpointResult, Location
from geonames.services.intent import LocationResolver, QuestionParser
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
        llm_client: LlmClient,
        fetchers: Mapping[str, Fetcher],
        parser: Optional[QuestionParser] = None,
        resolver: Optional[LocationResolver] = None,
        trace_collector: Optional[TraceCollector] = None,
    ) -> None:
        self._llm_client = llm_client
        self._fetchers = dict(fetchers)
        self._parser = parser or QuestionParser()
        self._resolver = resolver or LocationResolver()
        self.trace_collector = trace_collector or TestTraceCollector()

    def plan(self, question: str) -> AssistantPlan:
        """Parse the question without fetching or summarizing anything."""
        return self._parser.parse(question)

    def answer(self, question: str) -> str:
        """Answer the question: parse, fetch, summarize (with telemetry)."""
        self.trace_collector.reset()
        plan = self._parser.parse(question)
        location = self._resolver.resolve(plan.location)
        self.trace_collector.on_plan(asdict(plan))
        self.trace_collector.on_location_resolved(asdict(location))

        results: list[EndpointResult] = []
        for endpoint in plan.endpoints:
            params = _endpoint_params(endpoint, location, plan)
            self.trace_collector.on_tool_called(endpoint, params)
            data = self._fetchers[endpoint](params)
            results.append(EndpointResult(endpoint=endpoint, params=params, data=data))
        return self._llm_client.summarize(question, results, location=location.name)

    def process_user_query(self, user_input: str) -> str:
        """Golden-anchor contract alias for :meth:`answer`."""
        return self.answer(user_input)


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
    if endpoint == "earthquakes" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", plan.time_window):
        params["date"] = plan.time_window
    return params
