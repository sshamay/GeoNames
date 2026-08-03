"""Shared fixtures for the GeoNames test suite.

The test environment profile is loaded through the exact same code path as
production settings (geonames.config_loader.load_config) so tests never
re-declare runtime settings. One session-scoped client is shared; the service
fixtures wrap it so tests can call each endpoint through its own API class.
"""

from __future__ import annotations

import pytest

from geonames.clients import GeoNamesClient
from geonames.config_loader import Settings, load_config
from geonames.services import EarthquakesAPI, WeatherAPI
from test_utils.find_nearby import FindNearbyAPI
from test_utils.kpi_metrics import claim_mismatches


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings for the ``test`` environment, merged from config.yaml."""
    return load_config(env="test")


@pytest.fixture(scope="session")
def client(settings: Settings) -> GeoNamesClient:
    """Single shared GeoNamesClient built from the test settings."""
    return GeoNamesClient(
        username=settings.geonames_username,
        base_url=settings.geonames_base_url,
        timeout=settings.geonames_timeout,
    )


@pytest.fixture(scope="session")
def earthquakes_api(client: GeoNamesClient) -> EarthquakesAPI:
    """Earthquakes service API bound to the shared client."""
    return EarthquakesAPI(client)


@pytest.fixture(scope="session")
def find_nearby_api(client: GeoNamesClient) -> FindNearbyAPI:
    """FindNearby service API bound to the shared client."""
    return FindNearbyAPI(client)


@pytest.fixture(scope="session")
def weather_api(client: GeoNamesClient) -> WeatherAPI:
    """Weather service API bound to the shared client."""
    return WeatherAPI(client)


# =============================================================================
# PART 1A: GOLDEN ANCHOR EVAL (REQUIRED - copied from SmartSpend)
# =============================================================================

import importlib
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# Relative path (from this tests/ dir) to the golden anchor cases JSON.
_GOLDEN_ANCHOR_REL_PATH = os.path.join("data", "golden_anchor.json")

# Run KPIs: where the AQuA reporting ledger writes its reports.
_LEDGER_STASH_KEY = pytest.StashKey["AQuARunLedger"]()


def _golden_anchor_cases_path():
    here = os.path.dirname(__file__)
    return os.path.join(here, _GOLDEN_ANCHOR_REL_PATH)


def _load_golden_anchor_cases():
    """Load the golden anchor cases JSON."""
    cases_path = _golden_anchor_cases_path()
    if not os.path.exists(cases_path):
        raise FileNotFoundError(f"Golden anchor cases not found at {cases_path}")
    with open(cases_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def pytest_generate_tests(metafunc):
    """
    Dynamically parametrize golden anchor tests (one pytest case per JSON entry).
    """
    if "case_and_id" in metafunc.fixturenames:
        cases = _load_golden_anchor_cases()
        ids = [c.get("case_id") or str(i) for i, c in enumerate(cases)]
        metafunc.parametrize("case_and_id", cases, ids=ids)


def _load_aqua_evaluators():
    """Load the generic AQuAEvaluators class and register project hooks.

    ``hallucination_extractor`` wires the project's number-claim checker into
    the generic hallucination gate (reply numbers vs raw fetched data).
    """
    from test_utils.aqua_evaluation import AQuAEvaluators

    AQuAEvaluators.hallucination_extractor = staticmethod(claim_mismatches)
    return AQuAEvaluators


@pytest.fixture(scope="session")
def aqua_evaluators_class():
    """
    Load the AQuAEvaluators class from test_utils/aqua_evaluation.py.
    """
    return _load_aqua_evaluators()


def _render_aqua_dashboard(report_dir):
    """Regenerate reports/dashboard.html from the latest run + history.

    Loads scripts/render_dashboard.py via importlib so the repo scripts dir
    does not need to be importable (conftest runs from tests/). Never raises:
    the dashboard is best-effort and must not break a session.
    """
    here = os.path.dirname(__file__)
    script_path = os.path.join(os.path.dirname(here), "scripts", "render_dashboard.py")
    if not os.path.exists(script_path):
        return None
    spec = importlib.util.spec_from_file_location("render_dashboard", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render(
        report_dir=report_dir,
        out_path=os.path.join(report_dir, "dashboard.html"),
    )


@pytest.fixture(scope="session")
def golden_anchor_cases():
    """
    Load golden anchor test cases from JSON.
    """
    return _load_golden_anchor_cases()


@pytest.fixture(scope="session")
def golden_anchor_case_ids(golden_anchor_cases):
    """
    Generate readable case IDs for parametrized tests.
    """
    return [c.get("case_id") or str(i) for i, c in enumerate(golden_anchor_cases)]


@pytest.fixture(scope="session")
def run_ledger(request):
    """
    Session-scoped AQuA run ledger for KPI reporting.
    """
    from test_utils.aqua_reporting import AQuARunLedger

    ledger = AQuARunLedger()
    request.session.stash[_LEDGER_STASH_KEY] = ledger
    return ledger


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Track raw pytest outcomes + determinism (whole-suite context)."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        session = getattr(item, "session", None)
        ledger = session.stash.get(_LEDGER_STASH_KEY, None) if session is not None else None
        if ledger is not None:
            ledger.note_pytest_outcome(item.nodeid, report.outcome)
            # unit-marker tests are offline/deterministic; everything else (live
            # services, user flows, golden anchors) is probabilistic.
            ledger.note_determinism(bool(list(item.iter_markers("unit"))))


def pytest_sessionfinish(session, exitstatus):
    """
    GENERIC: flush the run ledger into reports/ at end of session.
    """
    ledger = session.stash.get(_LEDGER_STASH_KEY, None)
    if ledger is None or not ledger.entries:
        return
    ledger.duration_seconds = (time.time() - ledger.started_epoch)
    try:
        paths = ledger.write_reports(mutation_id=None)
        kpis = ledger.build_kpis()
        logger.info("AQuA KPI report written: %s", paths["run"])
        logger.info(
            "AQuA KPIs: %d cases | %d passed, %d failed, %d coverage gaps | "
            "pass_rate=%.1f%% escape_rate=%.1f%% mean_confidence=%.3f",
            kpis["totals"]["total"], kpis["totals"]["passed"], kpis["totals"]["failed"],
            kpis["totals"]["coverage_gaps"],
            100.0 * (kpis["pass_rate"] or 0.0), 100.0 * (kpis["escape_rate"] or 0.0),
            kpis["aggregate_confidence"]["mean"] or 0.0)
        try:
            dash_path = _render_aqua_dashboard(ledger.report_dir)
            if dash_path:
                logger.info("AQuA dashboard written: %s", dash_path)
        except Exception as dash_exc:  # pragma: no cover - dashboard is best-effort
            logger.warning("AQuA dashboard render failed: %s", dash_exc)
    except Exception as exc:  # pragma: no cover - reporter must never break a run
        logger.warning("AQuA KPI reporting failed: %s", exc)


# =============================================================================
# PART 2: PROJECT-SPECIFIC (GeoNames - the "Ask about a location" assistant)
# =============================================================================

@pytest.fixture(scope="session")
def assistant_class():
    """PROJECT-SPECIFIC: the GeoNames AskLocationAssistant class."""
    from geonames.services.ask_location import AskLocationAssistant

    return AskLocationAssistant


@pytest.fixture
def ai_assistant(assistant_class, settings):
    """PROJECT-SPECIFIC: assistant with production wiring decided by Settings.

    Uses build_assistant so the SUT owns its own composition (client, service
    fetchers, max_rows) instead of the test assembling it.
    """
    from geonames.factory import build_assistant

    assistant = build_assistant(settings)
    assert isinstance(assistant, assistant_class)
    return assistant
