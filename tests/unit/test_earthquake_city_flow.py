"""Unit tests for the earthquake-near-city flow (mocked service APIs).

The flow is pure orchestration, so fake API objects capture the exact params
the flow forwards - proving the "epicenter coords reused verbatim" contract
and the strongest-quake selection without any network access.
"""

import pytest

from test_utils.earthquake_city_flow import (
    EarthquakeCityFlowError,
    strongest_earthquake_near_city,
)

QUAKES = [
    {"eqid": "a", "datetime": "2023-02-06 01:17:00", "lat": 37.1, "lng": 37.0,
     "depth": 10, "src": "us", "magnitude": 6.4},
    {"eqid": "b", "datetime": "2023-02-06 02:00:00", "lat": 37.2, "lng": 37.1,
     "depth": 10, "src": "us", "magnitude": 7.8},
]

TOPYNYMS = [
    {"toponymName": "Atalar", "name": "Atalar", "distance": "0.93",
     "lat": "37.17", "lng": "37.03", "fcode": "PPL"},
]


class FakeEarthquakesAPI:
    def __init__(self, quakes):
        self.quakes = quakes
        self.last_params = None

    def get_earthquakes(self, **kwargs):
        self.last_params = kwargs
        return {"earthquakes": self.quakes}


class FakeFindNearbyAPI:
    def __init__(self, toponyms):
        self.toponyms = toponyms
        self.last_params = None

    def get_find_nearby(self, **kwargs):
        self.last_params = kwargs
        return {"geonames": self.toponyms}


@pytest.mark.unit
def test_flow_picks_strongest_and_forwards_verbatim_coords():
    earthquakes_api = FakeEarthquakesAPI(QUAKES)
    find_nearby_api = FakeFindNearbyAPI(TOPYNYMS)

    report = strongest_earthquake_near_city(
        earthquakes_api,
        find_nearby_api,
        bbox={"north": 39, "south": 35, "east": 39, "west": 35},
        min_magnitude=6.0,
        date="2023-02-06",
        radius=100,
    )

    assert report.quake["eqid"] == "b"
    assert report.report == "M7.8 near Atalar"

    assert find_nearby_api.last_params["lat"] == 37.2
    assert find_nearby_api.last_params["lng"] == 37.1
    assert find_nearby_api.last_params["radius"] == 100
    assert find_nearby_api.last_params["max_rows"] == 1


@pytest.mark.unit
def test_flow_filters_off_day_quakes_before_picking_strongest():
    """GeoNames ``date`` is "older or equal", so off-day rows must be dropped."""
    off_day = dict(QUAKES[0], datetime="2023-02-05 01:17:00", magnitude=9.0)
    earthquakes_api = FakeEarthquakesAPI([off_day, QUAKES[1]])

    report = strongest_earthquake_near_city(
        earthquakes_api,
        FakeFindNearbyAPI(TOPYNYMS),
        bbox={"north": 39, "south": 35, "east": 39, "west": 35},
        min_magnitude=6.0,
        date="2023-02-06",
        radius=100,
    )

    assert report.quake["eqid"] == "b"
    assert report.report == "M7.8 near Atalar"


@pytest.mark.unit
def test_flow_no_toponym_within_radius_yields_no_report():
    report = strongest_earthquake_near_city(
        FakeEarthquakesAPI(QUAKES),
        FakeFindNearbyAPI([]),
        bbox={"north": 39, "south": 35, "east": 39, "west": 35},
        min_magnitude=6.0,
        date="2023-02-06",
        radius=100,
    )

    assert report.toponym is None
    assert report.report is None


@pytest.mark.unit
def test_flow_raises_when_no_quakes_match():
    with pytest.raises(EarthquakeCityFlowError, match="no earthquakes"):
        strongest_earthquake_near_city(
            FakeEarthquakesAPI([]),
            FakeFindNearbyAPI(TOPYNYMS),
            bbox={"north": 39, "south": 35, "east": 39, "west": 35},
            min_magnitude=6.0,
            radius=100,
        )


@pytest.mark.unit
def test_flow_raises_when_date_filters_out_everything():
    off_day = dict(QUAKES[0], datetime="2023-01-01 01:17:00")
    with pytest.raises(EarthquakeCityFlowError, match="no earthquakes"):
        strongest_earthquake_near_city(
            FakeEarthquakesAPI([off_day]),
            FakeFindNearbyAPI(TOPYNYMS),
            bbox={"north": 39, "south": 35, "east": 39, "west": 35},
            min_magnitude=6.0,
            date="2023-02-06",
            radius=100,
        )
