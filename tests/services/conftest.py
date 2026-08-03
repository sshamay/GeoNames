"""Fixtures for the ground-truth catalog verification tests.

Only the FDSN reference-catalog clients are defined here; the GeoNames client
and service fixtures come from the root conftest.
"""

from __future__ import annotations

import pytest

from test_utils.fdsn_client import FdsnClient

USGS_BASE_URL = "https://earthquake.usgs.gov"
EMSC_BASE_URL = "https://www.seismicportal.eu"


@pytest.fixture(scope="session")
def usgs_api() -> FdsnClient:
    """FDSN event client bound to the USGS catalog."""
    return FdsnClient(USGS_BASE_URL)


@pytest.fixture(scope="session")
def emsc_api() -> FdsnClient:
    """FDSN event client bound to the EMSC catalog."""
    return FdsnClient(EMSC_BASE_URL)
