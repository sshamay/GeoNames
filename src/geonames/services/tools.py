"""LangChain tool definitions for the GeoNames agent.

Each ``@tool`` wraps a GeoNames service API and returns a compact dict the LLM
can reason about. A shared ``ExecutionContext`` is used to record every tool
invocation (tool name + parameters) into an AQuA-compatible trace collector.

Tool parameters are deliberately semantic (a place *name*), not geometric: the
LLM picks a place + which services to query, and the tools resolve the location
and bounding box internally. The telemetry records the resolved bounding-box
parameters so the AQuA agent_logic gate can still verify the request geometry.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from math import cos, radians
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from geonames.services.base import BaseAPI
from geonames.services.earthquakes import EarthquakesAPI
from geonames.services.search import SearchAPI
from geonames.services.weather import WeatherAPI

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_KM = 100.0


@dataclass
class ExecutionContext:
    """Per-query context shared by the agent and its tools.

    Records every tool call (name + parameters) and each tool's returned data,
    so the agent's wrapper can flush both into the ``trace_collector`` after
    the run (LangChain calls tools through the agent executor, so the tools
    emit into this context and the agent emits the final AQuA trace).
    """

    calls: List[Dict[str, Any]] = field(default_factory=list)
    outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record(self, name: str, parameters: Optional[Dict[str, Any]]) -> None:
        self.calls.append({"name": name, "parameters": parameters})

    def record_output(self, name: str, data: Dict[str, Any]) -> None:
        self.outputs[name] = data

    def reset(self) -> None:
        self.calls.clear()
        self.outputs.clear()


class GeoNamesTools:
    """Owner of the GeoNames LangChain tools (bound to real service APIs)."""

    def __init__(
        self,
        search_api: SearchAPI,
        earthquakes_api: EarthquakesAPI,
        weather_api: WeatherAPI,
        max_rows: int = 5,
        radius_km: float = DEFAULT_RADIUS_KM,
        default_location: Optional[str] = None,
    ) -> None:
        self._search = search_api
        self._earthquakes = earthquakes_api
        self._weather = weather_api
        self._max_rows = max_rows
        self._radius_km = radius_km
        self._default_location = default_location
        self.ctx = ExecutionContext()

    # -- location resolution ----------------------------------------------------

    def _resolve_place(self, place: str) -> str:
        """Map self-referential place phrases to the configured default location.

        Handles conversational inputs like "near me", "my location", or "here"
        that cannot be geocoded as-is: they refer to the user's own location,
        which the project resolves to ``default_location`` (e.g. Sacramento).
        Anything else is returned unchanged.
        """
        if not place:
            return place
        p = " ".join(place.lower().split())
        if p in {"near me", "my location", "my current location", "here", "near here"}:
            return self._default_location or place
        return place

    def _resolve(self, place: str) -> Dict[str, Any]:
        """Resolve a place name to lat/lng + bounding box. Raises on failure."""
        resp = self._search.search(q=place, max_rows=5)
        geonames = (resp or {}).get("geonames") or []
        if not geonames:
            raise LookupError(f"Could not geocode place: {place!r}")
        top = self._best_match(place, geonames)
        lat = float(top["lat"])
        lng = float(top["lng"])
        name = top.get("toponymName") or top.get("name") or place
        return {
            "name": name,
            "lat": lat,
            "lng": lng,
            "bbox": _bbox(lat, lng, self._radius_km),
        }

    @staticmethod
    def _best_match(place: str, geonames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pick the GeoNames result whose name best matches the query tokens.

        GeoNames' plain ``searchJSON`` ranks by popularity, so a bare
        "San Francisco" can resolve to the Honduran capital Tegucigalpa while
        the US city sits second. Prefer, in order: an exact-name match, a
        leading-name match ("Austin, TX" -> the city "Austin", not a metro
        area), then the most shared tokens (keeps "California" as the US
        region, not Tijuana).
        """
        q_full = place.lower().strip()
        q_lead = q_full.split(",")[0].strip()
        q_tokens = {t.lower() for t in re.findall(r"[a-z]+", q_full)}
        scored = []
        for i, geoname in enumerate(geonames):
            cand = geoname.get("toponymName") or geoname.get("name") or ""
            cand_l = cand.lower().strip()
            cand_tokens = {t.lower() for t in re.findall(r"[a-z]+", cand_l)}
            overlap = len(q_tokens & cand_tokens)
            exact = q_lead == cand_l
            leading = not exact and cand_l.startswith(q_lead)
            scored.append((exact, leading, overlap, geoname.get("fcl") == "P", -i, geoname))
        if not scored:
            return geonames[0]
        scored.sort(key=lambda s: (s[0], s[1], s[2], s[3], s[4]), reverse=True)
        return scored[0][5]

    # -- langchain tools ---------------------------------------------------------

    def geocode_location(self, place: str) -> Dict[str, Any]:
        """Resolve a city/place name to geographic coordinates + bounding box.

        Args:
            place: A human place name, e.g. "Sacramento" or "San Francisco".

        Returns:
            A dict with ``name``, ``lat``, ``lng``, and a ``bbox`` containing
            the north/south/east/west query box around the place.
        """
        place = self._resolve_place(place)
        self.ctx.record("geocode_location", {"place": place})
        result = self._resolve(place)
        self.ctx.record_output("geocode_location", result)
        return result
    def earthquakes(
        self,
        place: str,
        date: Optional[str] = None,
        max_rows: Optional[int] = None,
        max_distance_km: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fetch recent earthquakes near a city/place.

        The GeoNames ``date`` param is an *older-or-equal* cutoff (not an
        on-the-day filter), so to honor a caller's exact date we pass the user's
        date as the cutoff and then filter the returned quakes down to that
        exact day client-side. This turns the user's semantic "on <date>" into
        correct API behavior.

        ``max_distance_km`` narrows the quakes to those within ``max_distance_km``
        kilometers (straight-line) of the geocoded place center. Use it when the
        user names a radius (e.g. "within 50 miles" -> pass
        ``max_distance_km=80``; 1 mile ~= 1.609 km). When omitted, all quakes in
        the default search box are returned.

        Args:
            place: A human place name, e.g. "Sacramento".
            date: Optional ISO date (YYYY-MM-DD). Earthquakes are reduced to this
                exact day; when omitted, the most recent quakes are returned.
            max_rows: Optional cap on results (defaults to the configured cap).
            max_distance_km: Optional max straight-line distance in kilometers
                from the place center to filter returned quakes.

        Returns:
            A compact summary: count, strongest magnitude, and the list of quakes.
        """
        place = self._resolve_place(place)
        call_params = {"place": place, **({"date": date} if date else {})}
        if max_distance_km is not None:
            call_params["max_distance_km"] = max_distance_km
        self.ctx.record("earthquakes", call_params)
        location = self._resolve(place)
        params = dict(location["bbox"])
        data = self._earthquakes.get_earthquakes(
            **params, date=date, max_rows=max_rows or self._max_rows
        )
        quakes = data.get("earthquakes", []) or []
        if date:
            quakes = [q for q in quakes if str(q.get("datetime", ""))[:10] == date]
        if max_distance_km is not None:
            quakes = [
                q for q in quakes
                if _distance_km(location["lat"], location["lng"], q) <= max_distance_km
            ]
        # Record what the agent is actually told (post date-filter), so the
        # hallucination gate compares against the same set the reply describes,
        # not the broader raw fetch that a date filter narrowed down.
        telemetry = dict(data)
        telemetry["earthquakes"] = quakes
        self.ctx.record_output("earthquakes", telemetry)
        if not quakes:
            return (
                f"There are no recent earthquakes near {location['name']} "
                f"in the fetched data."
            )
        strongest = max(q["magnitude"] for q in quakes)
        strongest_dt = max(quakes, key=lambda q: q["magnitude"]).get("datetime", "unknown")
        summary = (
            f"There are {len(quakes)} recent earthquakes near {location['name']}. "
            f"The strongest earthquake has a magnitude of {strongest} on "
            f"{strongest_dt}. The earthquakes are: "
            + "; ".join(
                f"magnitude {q.get('magnitude')} on {q.get('datetime', 'unknown')}"
                for q in quakes
            )
        )
        return summary

    def weather(self, place: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """Fetch current weather observations near a city/place.

        Args:
            place: A human place name, e.g. "Sacramento".
            max_rows: Optional cap on results (defaults to the configured cap).

        Returns:
            A compact summary: station count and the observation list.
        """
        place = self._resolve_place(place)
        self.ctx.record("weather", {"place": place})
        location = self._resolve(place)
        params = dict(location["bbox"])
        data = self._weather.get_weather(**params, max_rows=max_rows or self._max_rows)
        observations = data.get("weatherObservations", []) or []
        self.ctx.record_output("weather", data)
        if not observations:
            return (
                f"There are no weather observations near {location['name']} "
                f"in the fetched data."
            )
        summary = (
            f"There are {len(observations)} weather observations near "
            f"{location['name']}. The observations are: "
            + "; ".join(
                f"{o.get('stationName', 'station')} at "
                f"{o.get('lng')},{o.get('lat')} temp {o.get('temperature')} C, "
                f"condition {o.get('weatherCondition', 'n/a')}"
                for o in observations
            )
        )
        return summary


def _bbox(lat: float, lng: float, radius_km: float) -> Dict[str, float]:
    """A square bounding box of ``radius_km`` around a centre point."""
    lat_delta = radius_km / 111.0
    lng_delta = radius_km / (111.0 * max(cos(radians(lat)), 0.05))
    return {
        "north": lat + lat_delta,
        "south": lat - lat_delta,
        "east": lng + lng_delta,
        "west": lng - lng_delta,
    }


def _distance_km(lat: float, lng: float, quake: Dict[str, Any]) -> float:
    """Approx straight-line distance (km) from (lat, lng) to a quake.

    Equirectangular projection, consistent with :func:`_bbox` (111 km/deg).
    """
    qlat = float(quake.get("lat") or 0.0)
    qlng = float(quake.get("lng") or 0.0)
    x = (qlng - lng) * cos(radians(lat))
    y = qlat - lat
    return 111.0 * ((x * x + y * y) ** 0.5)


def build_tools(api: GeoNamesTools) -> List[Any]:
    """Return the LangChain tool objects bound to a ``GeoNamesTools`` instance."""
    return [
        tool(api.geocode_location),
        tool(api.earthquakes),
        tool(api.weather),
    ]
