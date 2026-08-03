"""Unit tests for the ground-truth matching engine.

The matching module is pure (no network): time normalization, haversine
distance and greedy matching are locked down here with hand-built inputs.
"""

from datetime import datetime, timezone

import pytest

from test_utils.ground_truth_matching import (
    ReferenceEvent,
    _iso_to_epoch_seconds,
    _to_epoch_seconds,
    geonames_to_events,
    haversine_km,
    match_events,
    parse_emsc_events,
    parse_usgs_events,
)


@pytest.mark.unit
def test_haversine_known_distance():
    """Great-circle distance matches the classic LA-NY reference (~3936 km)."""
    distance = haversine_km(34.05, -118.24, 40.71, -74.01)
    assert 3900 < distance < 4000


@pytest.mark.unit
def test_haversine_zero_distance_for_same_point():
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == 0.0


@pytest.mark.unit
def test_geonames_datetime_parsed_as_utc():
    """GeoNames datetimes are naive UTC strings, not local time."""
    events = geonames_to_events(
        [{"eqid": "c1", "datetime": "2011-03-11 05:46:23", "lat": 38.3,
          "lng": 142.4, "magnitude": 8.8, "depth": 24.4}]
    )
    expected = 1299822383.0
    assert events[0].time == expected
    assert events[0].magnitude == 8.8


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2011-03-11T23:59:21.8Z", 1299887961.8),
        ("2011-03-11T10:43:09.3Z", 1299840189.3),
        ("2011-03-11T05:46:00Z", 1299822360.0),
    ],
)
def test_iso_time_parsing(value, expected):
    """ISO-8601 UTC timestamps with trailing Z and fractional seconds parse."""
    assert _iso_to_epoch_seconds(value) == expected


@pytest.mark.unit
def test_to_epoch_seconds_aware_datetime():
    """A tz-aware datetime is normalized to UTC before conversion."""
    aware = datetime(2011, 3, 11, 6, 46, 23, tzinfo=timezone.utc)
    assert _to_epoch_seconds(aware) == _to_epoch_seconds(
        datetime(2011, 3, 11, 6, 46, 23)
    )


def _ref(source, id, time, lat, lng, magnitude, depth=10.0):
    return ReferenceEvent(source, id, time, lat, lng, magnitude, depth)


@pytest.mark.unit
def test_parse_usgs_events_normalizes_epoch_ms():
    response = {
        "features": [
            {
                "id": "usp000hw62",
                "properties": {"mag": 5.4, "time": 1299887961080},
                "geometry": {"coordinates": [141.506, 36.528, 22.3]},
            }
        ]
    }
    events = parse_usgs_events(response)
    assert len(events) == 1
    assert events[0].id == "usp000hw62"
    assert events[0].time == 1299887961.080
    assert (events[0].lat, events[0].lng) == (36.528, 141.506)
    assert events[0].magnitude == 5.4


@pytest.mark.unit
def test_parse_emsc_events_normalizes_iso_time():
    response = {
        "features": [
            {
                "id": "20110311_0000181",
                "properties": {
                    "mag": 5.5,
                    "time": "2011-03-11T23:59:21.8Z",
                    "lat": 36.61,
                    "lon": 141.56,
                    "depth": 23.0,
                },
            }
        ]
    }
    events = parse_emsc_events(response)
    assert len(events) == 1
    assert events[0].id == "20110311_0000181"
    assert events[0].time == 1299887961.8
    assert (events[0].lat, events[0].lng) == (36.61, 141.56)


@pytest.mark.unit
def test_match_events_pairs_nearest_unused_reference():
    """Greedy matching never reuses a reference event for two GeoNames events."""
    geonames = [
        _ref("geonames", "g1", 100.0, 1.0, 1.0, 5.0),
        _ref("geonames", "g2", 200.0, 2.0, 2.0, 5.0),
    ]
    reference = [
        _ref("usgs", "r1", 100.0, 1.0, 1.0, 5.1),
        _ref("usgs", "r2", 200.0, 2.0, 2.0, 4.9),
    ]
    result = match_events(geonames, reference, time_tolerance_s=10, distance_km=10)
    assert result.coverage == 1.0
    assert [m.reference.id for m in result.matches] == ["r1", "r2"]
    assert result.unmatched_geonames == []
    assert result.unmatched_reference == []


@pytest.mark.unit
def test_match_events_skips_outside_tolerances():
    geonames = [
        _ref("geonames", "g1", 100.0, 1.0, 1.0, 5.0),
        _ref("geonames", "g2", 300.0, 1.0, 1.0, 5.0),
    ]
    reference = [_ref("usgs", "r1", 100.0, 1.0, 1.0, 5.0)]
    result = match_events(geonames, reference, time_tolerance_s=10, distance_km=10)
    assert result.coverage == 0.5
    assert [e.id for e in result.unmatched_geonames] == ["g2"]
    assert [m.geonames.id for m in result.matches] == ["g1"]


@pytest.mark.unit
def test_match_events_measures_magnitude_diff():
    geonames = [_ref("geonames", "g1", 100.0, 1.0, 1.0, 5.0)]
    reference = [_ref("usgs", "r1", 100.0, 1.0, 1.0, 5.4)]
    result = match_events(geonames, reference, time_tolerance_s=10, distance_km=10)
    assert result.matches[0].magnitude_diff == pytest.approx(0.4)
