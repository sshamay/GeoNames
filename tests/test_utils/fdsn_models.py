"""Pydantic models for validating FDSN event responses (USGS / EMSC).

Test-owned schemas mirroring the raw feature objects returned by each
catalog. USGS reports time as epoch milliseconds and coordinates as a
``[lon, lat, depth]`` triple; EMSC reports time as an ISO-8601 string with
lat/lon/depth as top-level properties. Extra fields returned by the catalogs
are tolerated (pydantic's default), so the models never reject a valid
response.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, field_validator


class UsgsProperties(BaseModel):
    """The ``properties`` object of a USGS event feature."""

    mag: Optional[float] = None
    time: Optional[int] = None
    magType: Optional[str] = None
    place: Optional[str] = None


class UsgsGeometry(BaseModel):
    """The ``geometry`` object of a USGS event feature."""

    coordinates: List[float]

    @field_validator("coordinates")
    @classmethod
    def coordinates_valid(cls, value: List[float]) -> List[float]:
        if len(value) != 3:
            raise ValueError("coordinates must be [lon, lat, depth]")
        lng, lat, _ = value
        if not -90 <= lat <= 90:
            raise ValueError("lat must be in [-90, 90]")
        if not -180 <= lng <= 180:
            raise ValueError("lng must be in [-180, 180]")
        return value


class UsgsFeature(BaseModel):
    """A single USGS event feature."""

    id: str
    properties: UsgsProperties
    geometry: UsgsGeometry


class UsgsResponse(BaseModel):
    """The full USGS GeoJSON feature collection."""

    features: List[UsgsFeature]


class EmscProperties(BaseModel):
    """The ``properties`` object of an EMSC event feature."""

    mag: Optional[float] = None
    time: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    depth: Optional[float] = None
    magtype: Optional[str] = None

    @field_validator("lat")
    @classmethod
    def lat_in_range(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not -90 <= value <= 90:
            raise ValueError("lat must be in [-90, 90]")
        return value

    @field_validator("lon")
    @classmethod
    def lon_in_range(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not -180 <= value <= 180:
            raise ValueError("lon must be in [-180, 180]")
        return value


class EmscFeature(BaseModel):
    """A single EMSC event feature."""

    id: str
    properties: EmscProperties


class EmscResponse(BaseModel):
    """The full EMSC feature collection."""

    features: List[EmscFeature]
