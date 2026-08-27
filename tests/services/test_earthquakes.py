"""Tests for the GeoNames earthquakes endpoint.

Data-driven: bounding boxes, filter cases and assertions come from JSON files
under tests/data/ (see aqua.loaders).
"""

from datetime import datetime
from typing import Any, Dict, List

import pytest

from geonames.clients import GeoNamesClient, GeoNamesClientError
from geonames.models import EarthquakesResponse
from aqua.loaders import load_cases

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_BBOX_KEYS = ("north", "south", "east", "west")


def _bbox(case: Dict[str, Any]) -> Dict[str, Any]:
    return {key: case[key] for key in _BBOX_KEYS}


def _quakes(api, **kwargs):
    return api.get_earthquakes(**kwargs)["earthquakes"]


@pytest.mark.services
@pytest.mark.parametrize(
    "case", load_cases("earthquake_bbox_cases.json"), ids=lambda c: c["id"]
)
def test_earthquakes_happy_path(earthquakes_api, case):
    """A bounding-box query returns exactly one well-formed quake."""
    response = EarthquakesResponse.model_validate(
        earthquakes_api.get_earthquakes(**_bbox(case), max_rows=1)
    )

    assert len(response.earthquakes) == 1


@pytest.mark.services
@pytest.mark.parametrize(
    "case", load_cases("earthquake_bbox_cases.json"), ids=lambda c: c["id"]
)
def test_earthquakes_response_schema(earthquakes_api, case):
    """Every row parses into the Earthquake schema with all documented fields."""
    response = EarthquakesResponse.model_validate(
        earthquakes_api.get_earthquakes(**_bbox(case), max_rows=5)
    )

    assert 1 <= len(response.earthquakes) <= 5
    for quake in response.earthquakes:
        assert quake.src
        assert quake.eqid


@pytest.mark.services
@pytest.mark.parametrize(
    "case", load_cases("earthquake_bbox_cases.json"), ids=lambda c: c["id"]
)
def test_earthquakes_data_integrity(earthquakes_api, case):
    """Magnitude, coordinates, datetime and depth are within valid ranges."""
    response = EarthquakesResponse.model_validate(
        earthquakes_api.get_earthquakes(**_bbox(case), max_rows=5)
    )

    for quake in response.earthquakes:
        assert 0 < quake.magnitude <= 10
        assert -90 <= quake.lat <= 90
        assert -180 <= quake.lng <= 180
        assert quake.depth >= 0
        assert datetime.strftime(quake.datetime, DATETIME_FORMAT)


@pytest.mark.services
@pytest.mark.parametrize(
    "case", load_cases("earthquake_filter_cases.json"), ids=lambda c: c["id"]
)
def test_earthquakes_filters(earthquakes_api, case):
    """API filters (magnitude/date/max_rows) and client-side depth filter."""
    bbox = {"north": 44.1, "south": -9.9, "east": -22.4, "west": 55.2}
    quakes = _quakes(earthquakes_api, **{**bbox, **case["params"]})

    assert len(quakes) >= 1
    for spec in case["asserts"]:
        _assert_filter(quakes, spec)


def _assert_filter(quakes: List[Dict[str, Any]], spec: Dict[str, Any]) -> None:
    """Apply one assertion spec from the filter test data."""
    operation, field, value = spec["op"], spec.get("field"), spec["value"]
    if operation == "min":
        assert all(q[field] >= value for q in quakes)
    elif operation == "max":
        assert all(q[field] <= value for q in quakes)
    elif operation == "row_count":
        assert len(quakes) <= value
    else:
        raise AssertionError(f"unknown filter assertion op: {operation}")


@pytest.mark.services
def test_earthquakes_rejects_unknown_username():
    """An unregistered username is rejected with a clear client error."""
    client = GeoNamesClient(username="not_a_real_geonames_user")

    with pytest.raises(GeoNamesClientError, match="401"):
        client.get(
            "earthquakesJSON",
            {"north": 44.1, "south": -9.9, "east": -22.4, "west": 55.2, "maxRows": 1},
        )
