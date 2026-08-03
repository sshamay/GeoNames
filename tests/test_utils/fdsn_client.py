"""Generic HTTP client for FDSN web-service event queries.

Used as the ground-truth reference for the GeoNames earthquake tests: the
same FDSN event protocol is served by both USGS and EMSC, so one generic
client is instantiated once per catalog with its own base URL. Transport-only,
in the same spirit as GeoNamesClient: it performs the GET and returns the raw
JSON response exactly as the service returned it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


class FdsnClientError(RuntimeError):
    """Raised when an FDSN event query fails."""


class FdsnClient:
    """Generic transport for FDSNWS event/1/query endpoints."""

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def query_events(
        self,
        starttime: str,
        endtime: str,
        min_latitude: float,
        max_latitude: float,
        min_longitude: float,
        max_longitude: float,
        min_magnitude: Optional[float] = None,
        format: str = "geojson",
        max_events: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Query events and return the raw JSON response body.

        Args:
            starttime/endtime: ISO 8601 UTC window, e.g. ``2011-03-11T00:00:00``.
            min/max latitude/longitude: bounding box for the query.
            min_magnitude: only events with magnitude >= this value.
            format: response format (``geojson`` for USGS, ``json`` for EMSC).
            max_events: cap on the number of returned events (``limit``).

        Returns:
            The decoded JSON response, exactly as returned by the catalog.

        Raises:
            FdsnClientError: On transport, HTTP or JSON decoding failure.
        """
        params: Dict[str, Any] = {
            "format": format,
            "starttime": starttime,
            "endtime": endtime,
            "minlatitude": min_latitude,
            "maxlatitude": max_latitude,
            "minlongitude": min_longitude,
            "maxlongitude": max_longitude,
        }
        if min_magnitude is not None:
            params["minmagnitude"] = min_magnitude
        if max_events is not None:
            params["limit"] = max_events

        url = f"{self.base_url}/fdsnws/event/1/query"
        logger.info("GET %s params=%s", url, params)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            logger.info("GET %s -> %s", url, response.status_code)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FdsnClientError(
                f"FDSN event query failed for {self.base_url}: {exc}"
            ) from exc
        try:
            return response.json()
        except ValueError as exc:
            raise FdsnClientError(
                f"FDSN event query returned non-JSON response from {self.base_url}"
            ) from exc
