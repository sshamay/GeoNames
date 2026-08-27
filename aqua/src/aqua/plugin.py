"""Pytest plugin for the AQuA framework (entry point: ``pytest11 = aqua.plugin``).

Provides everything the golden-anchor harness needs that is generic:

- Host config discovery: an ``aqua_config.py`` in the project root is loaded
  automatically as the one place to configure the framework (paths, judge,
  hallucination extractor). ``AQUA_*`` env vars and the adapter fixtures below
  are the fallbacks.
- ``--aqua-cases`` option + ``pytest_generate_tests`` parametrization of the
  ``case_and_id`` fixture (one pytest case per golden-anchor JSON entry).
- Session ``run_ledger`` fixture + reporting hooks (``pytest_runtest_makereport``
  tracks outcomes/determinism; ``pytest_sessionfinish`` writes the run reports
  and renders the dashboard).
- ``aqua_evaluators_class`` fixture wiring the judge + hallucination extractor
  into ``AQuAEvaluators``. Both are adapter fixtures the host project may
  override in conftest:
    - ``aqua_judge_config`` (default: ``aqua_config.JUDGE``, else ``AQUA_JUDGE_*``)
    - ``aqua_hallucination_extractor`` (default: ``aqua_config.hallucination_extractor``)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from types import ModuleType
from typing import Any, Dict, List, Optional

import pytest

from aqua.config import JudgeConfig, Thresholds, judge_config_from_env
from aqua.evaluation import AQuAEvaluators
from aqua.judge import build_llm_judge
from aqua.reporting import AQuARunLedger
from aqua import dashboard

logger = logging.getLogger(__name__)

_DEFAULT_CASES_PATH = "tests/data/golden_anchor.json"
_LEDGER_STASH_KEY = pytest.StashKey["AQuARunLedger"]()
_HOST_CONFIG_ENV = "AQUA_CONFIG"
_HOST_CONFIG_FILE = "aqua_config.py"


# ----------------------------------------------------------------- host config


def _load_host_config(config_path) -> Optional[ModuleType]:
    """Load the host project's ``aqua_config.py``, or None when absent.

    A plain file import (not package-based) so the host module's top-level
    code has no sys.path side effects on the rest of the suite.
    """
    if not config_path or not os.path.exists(config_path):
        return None
    spec = importlib.util.spec_from_file_location("aqua_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_judge_config(host: Optional[ModuleType]) -> JudgeConfig:
    """Judge settings: host ``JUDGE``/``JudgeConfig``, else env vars."""
    if host is not None:
        judge = getattr(host, "JUDGE", None) or getattr(host, "JudgeConfig", None)
        if isinstance(judge, JudgeConfig):
            return judge
    return judge_config_from_env()


def resolve_hallucination_extractor(host: Optional[ModuleType]):
    """Hallucination gate hook from the host config, or None when not set."""
    if host is None:
        return None
    extractor = getattr(host, "hallucination_extractor", None)
    return extractor if callable(extractor) else None


def resolve_thresholds(host: Optional[ModuleType]) -> Thresholds:
    """Evaluation thresholds: host ``THRESHOLDS``, else framework defaults."""
    if host is not None:
        thresholds = getattr(host, "THRESHOLDS", None)
        if isinstance(thresholds, Thresholds):
            return thresholds
    return Thresholds()


def pytest_addoption(parser):
    parser.addoption(
        "--aqua-cases",
        action="store",
        default=None,
        help=(
            "Path to the golden-anchor cases JSON (default: $AQUA_CASES or "
            f"{_DEFAULT_CASES_PATH!r})."
        ),
    )


def pytest_configure(config):
    """Register markers and discover the host ``aqua_config.py``."""
    config.addinivalue_line("markers", "ai_assistant: golden-anchor SUT test cases")
    config.addinivalue_line("markers", "unit: fast, isolated tests with mocks (no network)")

    host_path = os.environ.get(_HOST_CONFIG_ENV)
    if not host_path:
        rootdir = getattr(config, "rootdir", None) or config.invocation_dir
        host_path = os.path.join(str(rootdir), _HOST_CONFIG_FILE)
    host = _load_host_config(host_path)
    config._aqua_host_config = host
    if host is not None:
        # Path knobs from the host config become env defaults so the existing
        # loaders/reporting (which read AQUA_* at runtime) pick them up.
        for attr, env in (("CASES_PATH", "AQUA_CASES"),
                          ("DATA_DIR", "AQUA_DATA_DIR"),
                          ("REPORT_DIR", "AQUA_REPORT_DIR")):
            value = getattr(host, attr, None)
            if value:
                os.environ.setdefault(env, str(value))
        logger.debug("AQuA host config loaded from %s", host_path)


def _load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        cases = json.load(fh)
    if not isinstance(cases, list):
        raise pytest.UsageError(f"--aqua-cases file must contain a JSON list: {path}")
    return cases


def pytest_generate_tests(metafunc):
    """Parametrize one pytest case per golden-anchor JSON entry."""
    if "case_and_id" in metafunc.fixturenames:
        path = (
            metafunc.config.getoption("--aqua-cases")
            or os.environ.get("AQUA_CASES")
            or _DEFAULT_CASES_PATH
        )
        if not os.path.exists(path):
            raise pytest.UsageError(
                f"AQuA cases file not found: {path} (set --aqua-cases or AQUA_CASES)"
            )
        cases = _load_cases(path)
        ids = [c.get("case_id") or str(i) for i, c in enumerate(cases)]
        metafunc.parametrize("case_and_id", cases, ids=ids)


# ------------------------------------------------------------- adapter fixtures


@pytest.fixture
def aqua_judge_config(request):
    """Judge settings for the AQuA evaluation.

    Resolution order: the host's ``aqua_config.py`` ``JUDGE``/``JudgeConfig``,
    then ``AQUA_JUDGE_*`` env vars, then framework defaults. Host projects can
    still override this fixture in conftest for full control.
    """
    return resolve_judge_config(getattr(request.config, "_aqua_host_config", None))


@pytest.fixture
def aqua_llm_judge(aqua_judge_config):
    """The P6 judge hook; None keeps the suite offline/deterministic."""
    return build_llm_judge(aqua_judge_config)


@pytest.fixture
def aqua_hallucination_extractor(request):
    """Project hook: callable(ai_output, tool_outputs) -> list of number-claim
    mismatches. Taken from the host's ``aqua_config.py`` when provided; None
    (the default) disables the hallucination gate."""
    return resolve_hallucination_extractor(getattr(request.config, "_aqua_host_config", None))


@pytest.fixture
def aqua_thresholds(request):
    """Evaluation gate thresholds, applied to AQuAEvaluators.

    Resolution order: host ``aqua_config.py`` ``THRESHOLDS``, else framework
    defaults. Host projects can still override this fixture in conftest.
    """
    thresholds = resolve_thresholds(getattr(request.config, "_aqua_host_config", None))
    AQuAEvaluators.expected_outcome_semantic_threshold = thresholds.expected_outcome_semantic
    AQuAEvaluators.llm_judge_pass_threshold = thresholds.llm_judge_pass
    AQuAEvaluators.default_case_threshold = thresholds.default_case
    return thresholds


@pytest.fixture
def aqua_evaluators_class(aqua_llm_judge, aqua_hallucination_extractor, aqua_thresholds):
    """The AQuAEvaluators class with project hooks registered."""
    AQuAEvaluators.llm_judge = aqua_llm_judge
    AQuAEvaluators.hallucination_extractor = (
        staticmethod(aqua_hallucination_extractor)
        if aqua_hallucination_extractor is not None
        else None
    )
    return AQuAEvaluators


# ------------------------------------------------------------- run reporting


@pytest.fixture(scope="session")
def run_ledger(request):
    """Session-scoped run ledger for KPI reporting."""
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
            # unit-marker tests are offline/deterministic; everything else
            # (live services, user flows, golden anchors) is probabilistic.
            ledger.note_determinism(bool(list(item.iter_markers("unit"))))


def pytest_sessionfinish(session, exitstatus):
    """Flush the run ledger into reports/ at end of session."""
    ledger = session.stash.get(_LEDGER_STASH_KEY, None)
    if ledger is None or not ledger.entries:
        return
    ledger.duration_seconds = (time.time() - ledger.started_epoch)
    try:
        paths = ledger.write_reports(mutation_id=None)
        kpis = ledger.build_kpis()
        logger.info("Evaluation KPI report written: %s", paths["run"])
        logger.info(
            "Evaluation KPIs: %d cases | %d passed, %d failed, %d coverage gaps | "
            "pass_rate=%.1f%% escape_rate=%.1f%% mean_confidence=%.3f",
            kpis["totals"]["total"], kpis["totals"]["passed"], kpis["totals"]["failed"],
            kpis["totals"]["coverage_gaps"],
            100.0 * (kpis["pass_rate"] or 0.0), 100.0 * (kpis["escape_rate"] or 0.0),
            kpis["aggregate_confidence"]["mean"] or 0.0)
        try:
            dash_path = dashboard.render(
                report_dir=ledger.report_dir,
                out_path=os.path.join(ledger.report_dir, "dashboard.html"),
            )
            if dash_path:
                logger.info("Evaluation dashboard written: %s", dash_path)
        except Exception as dash_exc:  # pragma: no cover - dashboard is best-effort
            logger.warning("Evaluation dashboard render failed: %s", dash_exc)
    except Exception as exc:  # pragma: no cover - reporter must never break a run
        logger.warning("Evaluation KPI reporting failed: %s", exc)
