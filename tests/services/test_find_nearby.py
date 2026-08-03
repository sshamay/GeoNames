"""Tests for the GeoNames findNearby endpoint.

Data-driven: invalid coordinate cases come from tests/data/find_nearby_invalid_cases.json.
"""

import pytest

from geonames.models import StatusError
from test_utils.test_data import load_cases


@pytest.mark.services
@pytest.mark.parametrize(
    "case", load_cases("find_nearby_invalid_cases.json"), ids=lambda c: c["id"]
)
def test_find_nearby_rejects_invalid_coords(find_nearby_api, case):
    """Out-of-range coordinates return an API-level status error."""
    response = find_nearby_api.get_find_nearby(
        lat=case["lat"], lng=case["lng"], max_rows=1
    )

    status = StatusError.model_validate(response["status"])
    assert status.value == case["status_value"]
