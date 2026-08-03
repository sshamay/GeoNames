"""User flow: earthquake near a city (earthquakesJSON + findNearbyJSON).

For each case the strongest earthquake of a known day/bbox is located through
findNearby and rendered as an auto-report (e.g. ``M7.8 near Atalar``). Every
requirement from the flow spec is asserted against the raw data:

- both calls return HTTP 200 and bodies that validate against the response schemas
- the strongest quake has magnitude >= minMagnitude and lies inside the bbox
- the epicenter coordinates are reused verbatim in the findNearby call
- findNearby returns a name whose distance is within the radius
- the report is derived purely from the raw response data
"""

from __future__ import annotations

import pytest

from test_utils.earthquake_city_flow import strongest_earthquake_near_city
from geonames.models import Earthquake, Toponym
from test_utils.test_data import load_cases

CASES = load_cases("earthquake_city_cases.json")


@pytest.mark.user_flow
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_earthquake_near_city_report(flow_apis, case):
    """Full bbox -> strongest quake -> nearest city -> auto-report flow."""
    earthquakes_api, find_nearby_api, session = flow_apis

    report = strongest_earthquake_near_city(
        earthquakes_api,
        find_nearby_api,
        bbox=case["bbox"],
        min_magnitude=case["min_magnitude"],
        date=case["date"],
        radius=case["radius"],
        feature_class=case["feature_class"],
    )

    assert session.status_codes == [200, 200]

    quake = Earthquake.model_validate(report.quake)
    toponym = Toponym.model_validate(report.toponym)

    assert quake.magnitude >= case["min_magnitude"]
    assert case["bbox"]["south"] <= quake.lat <= case["bbox"]["north"]
    assert case["bbox"]["west"] <= quake.lng <= case["bbox"]["east"]

    find_nearby_call = session.find_calls("findNearbyJSON")[0]
    assert float(find_nearby_call.params["lat"]) == quake.lat
    assert float(find_nearby_call.params["lng"]) == quake.lng

    assert toponym.name or toponym.toponymName
    assert float(toponym.distance) <= case["radius"]

    name = toponym.toponymName or toponym.name
    assert report.report == f"M{quake.magnitude:.1f} near {name}"
