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
