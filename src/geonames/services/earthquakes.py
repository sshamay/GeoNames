"""Earthquakes service API (earthquakesJSON endpoint)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geonames.models.geonames import EarthquakesResponse
from geonames.services.base import BaseAPI, drop_none


class EarthquakesAPI(BaseAPI):
    """Recent earthquakes within a bounding box.

    Depth filtering is client-side: the API exposes no depth parameter, so
    min_depth/max_depth post-filter the (validated) results locally.
    """

    ENDPOINT = "earthquakesJSON"

    def get_earthquakes(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        date: Optional[str] = None,
        min_magnitude: Optional[float] = None,
        max_rows: Optional[int] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Query earthquakes, validating the raw response with Pydantic."""
        params = drop_none(
            {
                "north": north,
                "south": south,
                "east": east,
                "west": west,
                "date": date,
                "minMagnitude": min_magnitude,
                "maxRows": max_rows,
            }
        )
        response = self._request(params, EarthquakesResponse)
        return _filter_by_depth(response, min_depth, max_depth)


def _filter_by_depth(
    response: Dict[str, Any],
    min_depth: Optional[float],
    max_depth: Optional[float],
) -> Dict[str, Any]:
    """Post-filter earthquakes by depth range (client-side filter)."""
    if min_depth is None and max_depth is None:
        return response
    if "earthquakes" not in response:
        return response
    quakes = response["earthquakes"]
    if min_depth is not None:
        quakes = [q for q in quakes if q["depth"] >= min_depth]
    if max_depth is not None:
        quakes = [q for q in quakes if q["depth"] <= max_depth]
    response["earthquakes"] = quakes
    return response
