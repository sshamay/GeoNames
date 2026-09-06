"""GeoNames search/geocoding service (searchJSON endpoint).

The agent uses this to resolve a free-text place name ("Sacramento") to a
lat/lng centre point, which it then feeds to the earthquakes/weather tools.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from geonames.clients.geonames_client import GeoNamesClient
from geonames.models.geonames import SearchResponse
from geonames.services.base import BaseAPI, drop_none


class SearchAPI(BaseAPI):
    """Geocode a place name to GeoNames toponyms (lat/lng)."""

    ENDPOINT = "searchJSON"

    def search(
        self,
        q: str,
        max_rows: Optional[int] = None,
        feature_class: Optional[str] = None,
        country: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query GeoNames for a place and return the raw ``geonames`` list."""
        params = drop_none(
            {
                "q": q,
                "maxRows": max_rows,
                "featureClass": feature_class,
                "country": country,
            }
        )
        return self._request(params, SearchResponse)
