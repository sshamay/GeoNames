"""Unit tests for logging setup.

Shows the project's mocking convention: the ``mocker`` fixture from
pytest-mock, patching at the boundary of the external call (logging.basicConfig).
"""

import logging

import pytest

from geonames.logging_setup import setup_logging


@pytest.mark.unit
def test_setup_logging_configures_basic_config(mocker):
    """The chosen level is forwarded to the standard logging API."""
    basic_config = mocker.patch("logging.basicConfig")

    setup_logging(level="INFO")

    basic_config.assert_called_once()
    assert basic_config.call_args.kwargs["level"] == logging.INFO


@pytest.mark.unit
def test_setup_logging_rejects_unknown_level():
    """Invalid level names fail fast instead of silently ignoring the value."""
    with pytest.raises(ValueError, match="Unknown log level"):
        setup_logging(level="SHOUTING")
