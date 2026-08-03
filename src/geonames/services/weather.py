"""Weather service API (weatherJSON endpoint)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geonames.models.geonames import WeatherResponse
from geonames.services.base import BaseAPI, drop_none


class WeatherAPI(BaseAPI):
    """Weather stations with the most recent observation in a bounding box."""

    ENDPOINT = "weatherJSON"

    def get_weather(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query weather stations, validating the raw response with Pydantic."""
        params = drop_none(
            {
                "north": north,
                "south": south,
                "east": east,
                "west": west,
                "maxRows": max_rows,
            }
        )
        return self._request(params, WeatherResponse)
