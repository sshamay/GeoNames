"""FindNearby service API (findNearbyJSON endpoint).

Test-owned user-flow infrastructure: the assistant never calls findNearby;
only the earthquake-near-city user flow does, so the API class lives here
next to the flow helpers instead of in the SUT's service package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from geonames.models.geonames import FindNearbyResponse
from geonames.services.base import BaseAPI, drop_none


class FindNearbyAPI(BaseAPI):
    """Closest toponym(s) to a lat/lng point."""

    ENDPOINT = "findNearbyJSON"

    def get_find_nearby(
        self,
        lat: float,
        lng: float,
        radius: Optional[float] = None,
        feature_class: Optional[str] = None,
        feature_code: Optional[str] = None,
        max_rows: Optional[int] = None,
        style: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query nearby toponyms, validating the raw response with Pydantic."""
        params = drop_none(
            {
                "lat": lat,
                "lng": lng,
                "radius": radius,
                "featureClass": feature_class,
                "featureCode": feature_code,
                "maxRows": max_rows,
                "style": style,
            }
        )
        return self._request(params, FindNearbyResponse)
