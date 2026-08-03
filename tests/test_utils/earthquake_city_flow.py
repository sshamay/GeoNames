"""Cross-service flow: strongest earthquake near a city.

Orchestrates two GeoNames services into one user-facing workflow:

1. ``earthquakesJSON`` (bbox + minMagnitude) -> the strongest event is picked
2. ``findNearbyJSON`` (epicenter coords + radius) -> the nearest populated place

and renders a human-readable auto-report from the raw data, e.g.
``M7.8 near Atalar``. The flow is transport-only logic: it consumes the
service API objects and returns raw response data untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class EarthquakeCityFlowError(RuntimeError):
    """Raised when the flow cannot produce a report (e.g. no quakes found)."""


@dataclass
class EarthquakeCityReport:
    """Outcome of the flow: strongest quake, nearest place and the report."""

    quake: Dict[str, Any]
    toponym: Optional[Dict[str, Any]] = None
    report: Optional[str] = None


def strongest_earthquake_near_city(
    earthquakes_api,
    find_nearby_api,
    *,
    bbox: Dict[str, float],
    min_magnitude: float,
    radius: float,
    date: Optional[str] = None,
    feature_class: Optional[str] = None,
    max_quakes: int = 50,
) -> EarthquakeCityReport:
    """Run the flow and return the strongest quake, nearest place and report.

    Args:
        earthquakes_api: service API for the earthquakesJSON endpoint.
        find_nearby_api: service API for the findNearbyJSON endpoint.
        bbox: ``north``/``south``/``east``/``west`` bounding box.
        min_magnitude: only quakes with magnitude >= this value are considered.
        radius: search radius (km) for the nearest place around the epicenter.
        date: optional GeoNames ``date`` filter (``YYYY-MM-DD``); when given,
            returned quakes are restricted to that day before selecting the
            strongest (GeoNames ``date`` means "older or equal").
        feature_class: optional feature class filter for findNearby (e.g. ``P``
            for populated places).
        max_quakes: max rows fetched from the earthquakes endpoint.

    Returns:
        An EarthquakeCityReport with the raw strongest quake, the raw nearest
        toponym (``None`` if the radius contains no toponym) and the rendered
        auto-report (``None`` when no toponym was found).

    Raises:
        EarthquakeCityFlowError: When no quake meets the criteria.
    """
    quakes = earthquakes_api.get_earthquakes(
        **bbox,
        date=date,
        min_magnitude=min_magnitude,
        max_rows=max_quakes,
    )["earthquakes"]
    if date:
        quakes = [quake for quake in quakes if quake["datetime"].startswith(date)]
    if not quakes:
        raise EarthquakeCityFlowError(
            f"no earthquakes >= {min_magnitude} in {bbox} for date={date}"
        )

    strongest = max(quakes, key=lambda quake: quake["magnitude"])
    nearby = find_nearby_api.get_find_nearby(
        lat=strongest["lat"],
        lng=strongest["lng"],
        radius=radius,
        feature_class=feature_class,
        max_rows=1,
    )
    toponyms = nearby.get("geonames", [])
    toponym = toponyms[0] if toponyms else None
    return EarthquakeCityReport(
        quake=strongest,
        toponym=toponym,
        report=_format_report(strongest, toponym),
    )


def _format_report(
    quake: Dict[str, Any],
    toponym: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Render ``M<magnitude> near <city name>`` from the raw response data."""
    if toponym is None:
        return None
    name = toponym.get("toponymName") or toponym.get("name")
    if not name:
        return None
    return f"M{quake['magnitude']:.1f} near {name}"
