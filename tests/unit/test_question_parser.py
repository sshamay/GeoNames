"""Unit tests for the assistant's intent parsing and location resolution."""

import pytest

from geonames.models.assistant import Location
from geonames.services.intent import (
    DEFAULT_RADIUS_KM,
    KM_PER_MILE,
    LocationResolver,
    QuestionParser,
    UnknownIntentError,
    UnknownLocationError,
)

parser = QuestionParser()


@pytest.mark.unit
def test_parse_detects_both_endpoints_for_canonical_question():
    plan = parser.parse("Any recent earthquakes or bad weather near Sacramento?")
    assert plan.endpoints == ("earthquakes", "weather")
    assert plan.location == "Sacramento"
    assert plan.time_window == "recent"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "endpoints"),
    [
        ("Any recent earthquakes near Tokyo?", ("earthquakes",)),
        ("What is the weather like around Paris?", ("weather",)),
        ("Recent storms near London?", ("weather",)),
        ("Rain near New York?", ("weather",)),
    ],
)
def test_parse_detects_single_endpoint(question, endpoints):
    assert parser.parse(question).endpoints == endpoints


@pytest.mark.unit
def test_parse_extracts_radius_km():
    assert parser.parse("Earthquakes within 50 km near Sacramento?").radius_km == 50.0
    assert parser.parse("Earthquakes within 20 miles near Paris?").radius_km == pytest.approx(
        20 * KM_PER_MILE
    )
    assert parser.parse("Earthquakes near Sacramento?").radius_km == DEFAULT_RADIUS_KM


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Earthquakes in the last 24 hours near Tokyo?", "last_24h"),
        ("Earthquakes in the last 7 days near Tokyo?", "last_7d"),
        ("Earthquakes in the last 2 weeks near Tokyo?", "last_2w"),
        ("Weather today near Paris?", "today"),
        ("Earthquakes since 2023-02-06 near Tokyo?", "2023-02-06"),
        ("Recent earthquakes near Tokyo?", "recent"),
    ],
)
def test_parse_extracts_time_window(question, expected):
    assert parser.parse(question).time_window == expected


@pytest.mark.unit
def test_parse_raises_on_no_supported_endpoint():
    with pytest.raises(UnknownIntentError, match="no supported GeoNames endpoint"):
        parser.parse("Where is the nearest coffee shop near Paris?")


@pytest.mark.unit
def test_parse_raises_on_no_location():
    with pytest.raises(ValueError, match="no location mentioned"):
        parser.parse("Any recent earthquakes?")


@pytest.mark.unit
def test_resolver_returns_location_for_known_place():
    location = LocationResolver().resolve("Sacramento")
    assert isinstance(location, Location)
    assert location.name == "Sacramento"
    assert -122 <= location.lng <= -121
    assert 38 <= location.lat <= 39


@pytest.mark.unit
def test_resolver_is_case_insensitive():
    assert LocationResolver().resolve("  SACRAMENTO  ").name == "SACRAMENTO"


@pytest.mark.unit
def test_resolver_raises_for_unknown_place():
    with pytest.raises(UnknownLocationError, match="unknown location"):
        LocationResolver().resolve("Atlantis")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "expected_location", "expected_time_window"),
    [
        # Ambiguous location (defaults to most common US state)
        ("Any recent earthquakes near Springfield?", "Springfield", "recent"),
        ("Any recent earthquakes near Columbus?", "Columbus", "recent"),
        ("Any earthquakes near Austin yesterday?", "Austin", "yesterday"),
    ],
)
def test_parse_extracts_ambiguous_location(question, expected_location, expected_time_window):
    plan = parser.parse(question)
    assert plan.location == expected_location
    assert plan.time_window == expected_time_window


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "expected_location"),
    [
        ("Any recent earthquakes near Springfield, IL?", "Springfield, IL"),
        ("Weather near Austin, TX around downtown?", "Austin, TX"),
        ("Any earthquakes near San Francisco, CA yesterday?", "San Francisco, CA"),
    ],
)
def test_parse_extracts_city_state_format(question, expected_location):
    plan = parser.parse(question)
    assert plan.location == expected_location


@pytest.mark.unit
def test_parse_conversational_near_me_defaults_to_sacramento():
    plan = parser.parse("Show my local weather near me")
    assert plan.location == "Sacramento"
    assert plan.endpoints == ("weather",)


@pytest.mark.unit
def test_parse_conversational_close_to_downtown():
    plan = parser.parse("Any earthquakes close to downtown Los Angeles yesterday?")
    assert plan.location == "Los Angeles"
    assert plan.endpoints == ("earthquakes",)
    assert plan.time_window == "yesterday"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("location_name", "expected_coords"),
    [
        ("Springfield", (39.7817, -89.6501)),  # Springfield, IL
        ("Columbus", (39.9612, -82.9988)),  # Columbus, OH
        ("Austin", (30.2672, -97.7431)),  # Austin, TX
        ("Springfield, IL", (39.7817, -89.6501)),
    ],
)
def test_resolver_resolves_new_us_locations(location_name, expected_coords):
    location = LocationResolver().resolve(location_name)
    assert isinstance(location, Location)
    assert location.lat == pytest.approx(expected_coords[0], abs=0.001)
    assert location.lng == pytest.approx(expected_coords[1], abs=0.001)


@pytest.mark.unit
def test_resolver_normalizes_city_state_suffix():
    """City, ST should resolve even if only 'City' is registered."""
    loc = LocationResolver().resolve("Austin, TX")
    assert loc.lat == pytest.approx(30.2672, abs=0.001)
    assert loc.lng == pytest.approx(-97.7431, abs=0.001)
