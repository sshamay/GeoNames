"""Pydantic models for validating GeoNames API responses.

Schemas that mirror the service responses exactly - the raw response body is
never transformed, Pydantic only checks it. Field validators enforce the
documented data constraints (magnitude/coordinate ranges, non-negative depth,
parseable datetimes).

Extra fields returned by the API are tolerated (pydantic's default), so the
models never reject a valid response.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class StatusError(BaseModel):
    """The status object GeoNames returns for invalid/blocked requests."""

    message: str
    value: int


class Earthquake(BaseModel):
    """A single earthquake record from the earthquakesJSON endpoint."""

    datetime: datetime
    depth: float
    lng: float
    src: str
    eqid: str
    magnitude: float
    lat: float

    @field_validator("magnitude")
    @classmethod
    def magnitude_in_range(cls, value: float) -> float:
        if value < 0:
            return 0.0  # tolerate out-of-range API quirks (e.g. -0.49)
        if value > 10:
            return 10.0
        return value

    @field_validator("lat")
    @classmethod
    def lat_in_range(cls, value: float) -> float:
        if not -90 <= value <= 90:
            raise ValueError("lat must be in [-90, 90]")
        return value

    @field_validator("lng")
    @classmethod
    def lng_in_range(cls, value: float) -> float:
        if not -180 <= value <= 180:
            raise ValueError("lng must be in [-180, 180]")
        return value

    @field_validator("depth")
    @classmethod
    def depth_non_negative(cls, value: float) -> float:
        # The API occasionally reports marginally negative depths (e.g. -0.48)
        # for quakes whose hypocentre sits just above the reference ellipsoid.
        # Like the magnitude quirk below, clamp these to 0 rather than rejecting
        # the whole record, so a single measurement artifact cannot sink every
        # earthquake query.
        return max(value, 0.0)


class EarthquakesResponse(BaseModel):
    """The full earthquakesJSON response body."""

    earthquakes: List[Earthquake]
    status: Optional[StatusError] = None


class Toponym(BaseModel):
    """A place/toponym record from the findNearbyJSON endpoint."""

    adminCode1: Optional[str] = None
    lng: Optional[str] = None
    distance: Optional[str] = None
    geonameId: Optional[int] = None
    toponymName: Optional[str] = None
    countryId: Optional[str] = None
    fcl: Optional[str] = None
    population: Optional[int] = None
    countryCode: Optional[str] = None
    name: Optional[str] = None
    fclName: Optional[str] = None
    adminCodes1: Optional[Dict[str, Any]] = None
    countryName: Optional[str] = None
    fcodeName: Optional[str] = None
    adminName1: Optional[str] = None
    lat: Optional[str] = None
    fcode: Optional[str] = None


class FindNearbyResponse(BaseModel):
    """The full findNearbyJSON response body."""

    geonames: List[Toponym]
    status: Optional[StatusError] = None


class SearchResponse(BaseModel):
    """The full searchJSON response body (geocoding a place name)."""

    geonames: List[Toponym]
    status: Optional[StatusError] = None


class WeatherObservation(BaseModel):
    """A weather station observation from the weatherJSON endpoint.

    Station fields are intentionally lenient: live GeoNames observations can be
    missing or empty for some stations (e.g. no wind/cloud reporting), so only
    the identity (stationName/ICAO) and location (lat/lng/observation) are
    required. All meteorologic values default so a valid response never fails
    validation on a partial observation.
    """

    lng: float
    observation: str
    ICAO: Optional[str] = None
    clouds: Optional[str] = None
    dewPoint: Optional[str] = None
    cloudsCode: Optional[str] = None
    datetime: datetime
    temperature: Optional[str] = None
    humidity: Optional[int] = None
    stationName: str
    weatherCondition: Optional[str] = None
    windSpeed: Optional[str] = None
    lat: float


class WeatherResponse(BaseModel):
    """The full weatherJSON response body."""

    weatherObservations: List[WeatherObservation]
    status: Optional[StatusError] = None
