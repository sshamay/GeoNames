"""
Golden Anchor testing for the GeoNames assistant.

The golden-anchor cases are split at the DATA level into two homogeneous
JSON files that both live under tests/data/:

1. ``tests/data/golden_anchor.json``            -> 18 ``assistant`` cases
   These verify the REAL assistant behavior and MUST PASS (e.g. GN-001 queries
   both tools, GN-025 maps "near me" -> Sacramento).

2. ``tests/data/golden_anchor_detector_tests.json`` -> 9 ``aqua-detector`` cases
   These do NOT test the assistant; they exercise the AQuA evaluation
   machinery itself (content_rules, agent_logic, hallucination_check, HITL).
   They are designed to be REJECTED by the detectors and are asserted to be
   rejected (i.e. the detector fires). A pass here means the guard is broken.

This file uses a SINGLE parametrized test function backed by the framework's
one ``case_and_id`` fixture (loader: ``--aqua-cases`` / ``AQUA_CASES``). Point
it at the file you want to run:

    # assistant behavior
    pytest tests/ai_assistant/test_golden_anchor_eval.py \
        --aqua-cases tests/data/golden_anchor.json

    # AQuA-detector machinery
    pytest tests/ai_assistant/test_golden_anchor_eval.py \
        --aqua-cases tests/data/golden_anchor_detector_tests.json

Each file is homogeneous (one category), so no cross-category skipping is
needed; the assertion below branches on the case's ``category``.
"""

import pytest

from aqua.golden_anchor import assert_golden_anchor, run_golden_anchor_case

pytestmark = pytest.mark.ai_assistant


def test_golden_anchor_case(ai_assistant, aqua_evaluators_class, run_ledger, case_and_id):
    """Run one golden-anchor case and assert its category-specific outcome."""
    case = case_and_id

    eval_result, ai_output = run_golden_anchor_case(
        ai_assistant, aqua_evaluators_class, run_ledger, case
    )

    category = case.get("category", "assistant")
    if category == "aqua-detector":
        # The AQuA detector demo must be flagged unsafe - i.e. rejected.
        case_id = case.get("case_id") or "unknown"
        assert eval_result.get("is_safe") is not True, (
            f"AQuA detector case {case_id} was NOT flagged - the guard failed to fire: "
            f"{eval_result.get('action')} aggregate={eval_result.get('aggregate_score')}"
        )
        return

    # Default: assistant behavior must pass.
    assert_golden_anchor(eval_result, ai_output)
