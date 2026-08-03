"""Ground-truth verification of GeoNames earthquakes against reference catalogs.

Every case is a well-known historic earthquake day. The GeoNames
``earthquakesJSON`` results for that day/bbox are matched against the USGS
(Gold Standard) and EMSC FDSN catalogs: each GeoNames event is paired with the
nearest reference event within configurable time and distance tolerances, then
the pair's magnitudes are cross-checked. Coverage below ``min_coverage`` and
magnitude differences above ``magnitude_tolerance`` are reported as failures,
surfacing critical gaps between GeoNames and the reference catalogs.

GeoNames ``date`` means "older or equal", so events are filtered to the case
day before matching. The reference catalogs are queried with a slightly lower
``minmagnitude`` so a GeoNames event is never falsely unmatched just because
the catalogs disagree on the last decimal (e.g. 5.0 vs 4.9). Magnitude
tolerance is per-case: mb-vs-mwc differences on large aftershocks can reach
~1.1 (Tohoku 2011 is calibrated to 1.2), while smaller-magnitude cases are
tighter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from test_utils.ground_truth_matching import (
    MatchResult,
    ReferenceEvent,
    geonames_to_events,
    match_events,
    parse_emsc_events,
    parse_usgs_events,
)
from test_utils.test_data import load_cases

logger = logging.getLogger(__name__)

BBOX_KEYS = ("north", "south", "east", "west")
CASES = load_cases("ground_truth_cases.json")
REFERENCE_MIN_MAGNITUDE_OFFSET = 0.5


def _bbox(case: Dict[str, Any]) -> Dict[str, Any]:
    return {key: case["bbox"][key] for key in BBOX_KEYS}


def _fdsn_bbox(case: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the GeoNames bbox naming to FDSN min/max lat/lon params."""
    bbox = case["bbox"]
    return {
        "min_latitude": bbox["south"],
        "max_latitude": bbox["north"],
        "min_longitude": bbox["west"],
        "max_longitude": bbox["east"],
    }


def _day_quakes(api, case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GeoNames earthquakes restricted to the case's date day."""
    quakes = api.get_earthquakes(
        **_bbox(case),
        date=case["date"],
        min_magnitude=case["min_magnitude"],
        max_rows=case["max_rows"],
    )["earthquakes"]
    return [quake for quake in quakes if quake["datetime"].startswith(case["date"])]


def _day_window(case: Dict[str, Any]) -> tuple[str, str]:
    date = case["date"]
    return f"{date}T00:00:00", f"{date}T23:59:59"


def _reference_min_magnitude(case: Dict[str, Any]) -> float:
    """Query the catalog a bit lower than GeoNames to avoid edge false misses."""
    return max(case["min_magnitude"] - REFERENCE_MIN_MAGNITUDE_OFFSET, 0.0)


def _verify_match_result(
    result: MatchResult,
    case: Dict[str, Any],
    reference_name: str,
) -> None:
    """Assert coverage and per-pair magnitude tolerance, surfacing gaps."""
    coverage = result.coverage
    unmatched = [event.id for event in result.unmatched_geonames]
    assert coverage >= case["min_coverage"], (
        f"{reference_name} coverage {coverage:.0%} < {case['min_coverage']:.0%}; "
        f"unmatched GeoNames events: {unmatched}"
    )
    for match in result.matches:
        assert match.magnitude_diff <= case["magnitude_tolerance"], (
            f"magnitude mismatch for {match.geonames.id}: "
            f"geonames={match.geonames.magnitude} vs "
            f"{reference_name}={match.reference.magnitude} "
            f"(diff {match.magnitude_diff:.2f})"
        )
    if result.unmatched_geonames:
        logger.warning(
            "%s: %d unmatched GeoNames events: %s",
            reference_name,
            len(result.unmatched_geonames),
            unmatched,
        )


@pytest.mark.services
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_geonames_matches_usgs(earthquakes_api, usgs_api, case):
    """GeoNames earthquakes are present in USGS with matching magnitudes."""
    quakes = _day_quakes(earthquakes_api, case)
    if not quakes:
        pytest.skip(f"no GeoNames earthquakes for {case['id']}")

    starttime, endtime = _day_window(case)
    reference = parse_usgs_events(
        usgs_api.query_events(
            starttime,
            endtime,
            **_fdsn_bbox(case),
            min_magnitude=_reference_min_magnitude(case),
        )
    )
    result = match_events(
        geonames_to_events(quakes),
        reference,
        case["time_tolerance_s"],
        case["distance_km"],
    )
    _verify_match_result(result, case, "usgs")


@pytest.mark.services
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_geonames_matches_emsc(earthquakes_api, emsc_api, case):
    """GeoNames earthquakes are present in EMSC with matching magnitudes."""
    quakes = _day_quakes(earthquakes_api, case)
    if not quakes:
        pytest.skip(f"no GeoNames earthquakes for {case['id']}")

    starttime, endtime = _day_window(case)
    reference = parse_emsc_events(
        emsc_api.query_events(
            starttime,
            endtime,
            **_fdsn_bbox(case),
            min_magnitude=_reference_min_magnitude(case),
            format="json",
        )
    )
    result = match_events(
        geonames_to_events(quakes),
        reference,
        case["time_tolerance_s"],
        case["distance_km"],
    )
    _verify_match_result(result, case, "emsc")
