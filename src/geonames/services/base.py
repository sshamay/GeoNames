"""Base class for GeoNames service APIs.

A service wraps the generic GeoNamesClient for one endpoint: it builds the
query params, delegates the GET to the generic client, validates the raw
response against a Pydantic model, and returns the response untouched.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from geonames.clients.geonames_client import GeoNamesClient
from geonames.models.geonames import StatusError


def drop_none(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values so optional params are simply omitted."""
    return {key: value for key, value in params.items() if value is not None}


class BaseAPI:
    """Base class for one GeoNames service endpoint."""

    ENDPOINT: str = ""

    def __init__(self, client: GeoNamesClient) -> None:
        self._client = client

    def _request(
        self,
        params: Dict[str, Any],
        model: Type,
    ) -> Dict[str, Any]:
        """GET the endpoint, validate the raw response, return it unchanged."""
        response = self._client.get(self.ENDPOINT, params)
        if "status" in response:
            StatusError.model_validate(response["status"])
        else:
            model.model_validate(response)
        return response
