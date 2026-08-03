"""Fixtures for user-flow tests.

User flows orchestrate multiple GeoNames services, so their APIs are bound to
a dedicated client using a recording session - the flow test can then assert
transport-level facts (HTTP statuses, exact query params) without affecting
the shared session-scoped client from the root conftest.
"""

from __future__ import annotations

import pytest

from geonames.clients import GeoNamesClient
from geonames.services import EarthquakesAPI
from test_utils.find_nearby import FindNearbyAPI
from test_utils.recording_session import RecordingSession


@pytest.fixture
def flow_apis(settings):
    """A (earthquakes_api, find_nearby_api, recording_session) triple."""
    session = RecordingSession()
    client = GeoNamesClient(
        username=settings.geonames_username,
        base_url=settings.geonames_base_url,
        timeout=settings.geonames_timeout,
        session=session,
    )
    return (
        EarthquakesAPI(client),
        FindNearbyAPI(client),
        session,
    )
