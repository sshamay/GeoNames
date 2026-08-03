"""Ground-truth matching between GeoNames earthquakes and reference catalogs.

GeoNames reports earthquake rows with a ``datetime`` string, coordinates,
magnitude and an ``eqid``. USGS/EMSC FDSN responses use different time and
coordinate layouts, so every event is first normalized to a common
:class:`ReferenceEvent`, then GeoNames events are matched to the nearest
reference event within configurable time and distance tolerances.

Matching is greedy: each GeoNames event is paired with the closest unused
reference event, so a single reference event is never double-counted.
"""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional, Tuple

from test_utils.fdsn_models import EmscResponse, UsgsResponse

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
GEONAMES_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class ReferenceEvent:
    """A normalized earthquake event, comparable across all sources."""

    source: str
    id: str
    time: float  # epoch seconds (UTC)
    lat: float
    lng: float
    magnitude: float
    depth: float


@dataclass
class EventMatch:
    """A matched (GeoNames, reference) event pair."""

    geonames: ReferenceEvent
    reference: ReferenceEvent
    time_diff_s: float
    distance_km: float
    magnitude_diff: float


@dataclass
class MatchResult:
    """Outcome of matching GeoNames events against a reference catalog."""

    matches: List[EventMatch]
    unmatched_geonames: List[ReferenceEvent]
    unmatched_reference: List[ReferenceEvent]

    @property
    def coverage(self) -> float:
        """Fraction of GeoNames events that matched a reference event."""
        if not self.matches:
            return 0.0
        total = len(self.matches) + len(self.unmatched_geonames)
        return len(self.matches) / total


def geonames_to_events(quakes: List[Dict[str, Any]]) -> List[ReferenceEvent]:
    """Normalize raw GeoNames earthquake rows into reference events."""
    events: List[ReferenceEvent] = []
    for quake in quakes:
        events.append(
            ReferenceEvent(
                source="geonames",
                id=quake["eqid"],
                time=_geonames_to_epoch_seconds(quake["datetime"]),
                lat=float(quake["lat"]),
                lng=float(quake["lng"]),
                magnitude=float(quake["magnitude"]),
                depth=float(quake["depth"]),
            )
        )
    return events


def parse_usgs_events(response: Dict[str, Any]) -> List[ReferenceEvent]:
    """Normalize a raw USGS GeoJSON response into reference events."""
    parsed = UsgsResponse.model_validate(response)
    events: List[ReferenceEvent] = []
    for feature in parsed.features:
        properties, geometry = feature.properties, feature.geometry
        if properties.mag is None or properties.time is None:
            logger.warning("USGS feature %s missing mag/time, skipping", feature.id)
            continue
        lng, lat, depth = geometry.coordinates
        events.append(
            ReferenceEvent(
                source="usgs",
                id=feature.id,
                time=properties.time / 1000.0,
                lat=lat,
                lng=lng,
                magnitude=properties.mag,
                depth=depth,
            )
        )
    return events


def parse_emsc_events(response: Dict[str, Any]) -> List[ReferenceEvent]:
    """Normalize a raw EMSC feature collection into reference events."""
    parsed = EmscResponse.model_validate(response)
    events: List[ReferenceEvent] = []
    for feature in parsed.features:
        properties = feature.properties
        if (
            properties.mag is None
            or properties.time is None
            or properties.lat is None
            or properties.lon is None
        ):
            logger.warning("EMSC feature %s missing mag/time/lat/lon, skipping", feature.id)
            continue
        events.append(
            ReferenceEvent(
                source="emsc",
                id=feature.id,
                time=_iso_to_epoch_seconds(properties.time),
                lat=properties.lat,
                lng=properties.lon,
                magnitude=properties.mag,
                depth=properties.depth if properties.depth is not None else 0.0,
            )
        )
    return events


def match_events(
    geonames: List[ReferenceEvent],
    reference: List[ReferenceEvent],
    time_tolerance_s: float,
    distance_km: float,
) -> MatchResult:
    """Greedily pair each GeoNames event with the nearest unused reference event.

    Args:
        geonames: normalized GeoNames events to explain.
        reference: normalized reference-catalog events.
        time_tolerance_s: maximum allowed origin-time difference in seconds.
        distance_km: maximum allowed epicentral distance in km.

    Returns:
        A MatchResult with the pairs, both unmatched lists and coverage.
    """
    matches: List[EventMatch] = []
    unmatched_geonames: List[ReferenceEvent] = []
    used = set()

    ordered = sorted(geonames, key=lambda event: event.time)
    for event in ordered:
        best: Optional[Tuple[float, float, int, ReferenceEvent]] = None
        for index, ref in enumerate(reference):
            if index in used:
                continue
            time_diff = abs(event.time - ref.time)
            if time_diff > time_tolerance_s:
                continue
            distance = haversine_km(event.lat, event.lng, ref.lat, ref.lng)
            if distance > distance_km:
                continue
            key = (time_diff, distance, index)
            if best is None or key < best[:3]:
                best = (time_diff, distance, index, ref)
        if best is None:
            unmatched_geonames.append(event)
            continue
        time_diff, distance, index, ref = best
        used.add(index)
        matches.append(
            EventMatch(
                geonames=event,
                reference=ref,
                time_diff_s=time_diff,
                distance_km=distance,
                magnitude_diff=abs(event.magnitude - ref.magnitude),
            )
        )

    unmatched_reference = [ref for index, ref in enumerate(reference) if index not in used]
    return MatchResult(
        matches=matches,
        unmatched_geonames=unmatched_geonames,
        unmatched_reference=unmatched_reference,
    )


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two points, in kilometres."""
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))
    dlat, dlng = lat2 - lat1, lng2 - lng1
    haversine = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(haversine))


def _geonames_to_epoch_seconds(value: str) -> float:
    """Parse a GeoNames ``datetime`` string as UTC epoch seconds."""
    dt = datetime.strptime(value, GEONAMES_DATETIME_FORMAT)
    return _to_epoch_seconds(dt)


def _iso_to_epoch_seconds(value: str) -> float:
    """Parse an ISO-8601 UTC timestamp (trailing ``Z``) as epoch seconds."""
    base = value[:-1] if value.endswith("Z") else value
    fraction = ""
    if "." in base:
        base, fraction = base.split(".", 1)
    dt = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S")
    fraction_seconds = float("0." + fraction) if fraction else 0.0
    return _to_epoch_seconds(dt) + fraction_seconds


def _to_epoch_seconds(dt: datetime) -> float:
    """Convert a naive/aware datetime to UTC epoch seconds."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return calendar.timegm(dt.timetuple()) + dt.microsecond / 1_000_000.0
