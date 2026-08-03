"""Generic HTTP client for the GeoNames web services.

Transport-only: knows nothing about individual endpoints. It appends the
username to every request (required by the API), performs the GET, and
returns the raw JSON response body exactly as the service returned it -
schema/data checking is the responsibility of the service layer and the
Pydantic response models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://secure.geonames.org"
DEFAULT_TIMEOUT = 30.0


class GeoNamesClientError(RuntimeError):
    """Raised when a GeoNames API request fails."""


class GeoNamesClient:
    """Generic transport for the GeoNames web services."""

    def __init__(
        self,
        username: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.username = username
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET an endpoint and return the raw JSON response body.

        Args:
            endpoint: API endpoint path, e.g. ``earthquakesJSON``.
            params: Query parameters (username is appended automatically).

        Returns:
            The decoded JSON response, exactly as returned by the service.

        Raises:
            GeoNamesClientError: On transport, HTTP or JSON decoding failure.
        """
        request_params = {**(params or {}), "username": self.username}
        url = f"{self.base_url}/{endpoint}"
        logger.info(
            "GET %s params=%s",
            url,
            {k: v for k, v in request_params.items() if k != "username"},
        )
        try:
            response = self.session.get(url, params=request_params, timeout=self.timeout)
            logger.info("GET %s -> %s", url, response.status_code)
            if response.status_code == 401:
                raise GeoNamesClientError(
                    f"{endpoint} rejected username '{self.username}' (401): "
                    "enable web services for the account at geonames.org"
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GeoNamesClientError(f"{endpoint} request failed: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise GeoNamesClientError(f"{endpoint} returned non-JSON response") from exc
