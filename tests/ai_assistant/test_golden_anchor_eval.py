"""
Generic Golden Anchor Testing Framework (host-project test file).

The generic evaluation logic lives in the ``aqua`` framework
(``aqua.golden_anchor``); this file wires it to the GeoNames assistant through
the AQuA adapter fixtures (``ai_assistant``, ``aqua_evaluators_class``,
``run_ledger``, ``case_and_id``) and registers the ``ai_assistant`` pytest
marker used by the run ledger's determinism tracking.

To use in another project: provide the same fixtures in your conftest
(an assistant exposing ``process_user_query`` + ``trace_collector``, and the
``aqua_judge_config`` / ``aqua_hallucination_extractor`` adapters) and point
pytest at your golden-anchor cases JSON via ``--aqua-cases``.
"""

import pytest

from aqua.golden_anchor import assert_golden_anchor, run_golden_anchor_case

pytestmark = pytest.mark.ai_assistant


def test_golden_anchor_case(ai_assistant, aqua_evaluators_class, run_ledger, case_and_id):
    """Run one golden anchor case and assert it passed (skip on coverage gaps)."""
    eval_result, ai_output = run_golden_anchor_case(
        ai_assistant, aqua_evaluators_class, run_ledger, case_and_id
    )
    assert_golden_anchor(eval_result, ai_output)
