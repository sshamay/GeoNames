"""Generic golden-anchor test logic, host-project agnostic.

The project's golden-anchor test file imports these two helpers:

.. code-block:: python

    import pytest
    from aqua.golden_anchor import assert_golden_anchor, run_golden_anchor_case

    pytestmark = pytest.mark.ai_assistant

    def test_golden_anchor_case(ai_assistant, aqua_evaluators_class, run_ledger, case_and_id):
        eval_result, ai_output = run_golden_anchor_case(
            ai_assistant, aqua_evaluators_class, run_ledger, case_and_id)
        assert_golden_anchor(eval_result, ai_output)

The fixtures are provided by the host project (``ai_assistant`` = the system
under test) and by the ``aqua.plugin`` pytest plugin (``aqua_evaluators_class``,
``run_ledger``, ``case_and_id``). The project registers its cases via the
``--aqua-cases`` pytest option (default ``$AQUA_CASES`` or
``tests/data/golden_anchor.json``).
"""

from __future__ import annotations

import pytest

from aqua.evaluation import extract_assistant_output, failure_summary, format_eval_result


def run_golden_anchor_case(ai_assistant, evaluators_class, run_ledger, case):
    """Run one golden-anchor case against the SUT and record the outcome.

    Args:
        ai_assistant: The system under test; must expose
            ``process_user_query(input)`` and ``trace_collector.get_trace_logs()``.
        evaluators_class: ``AQuAEvaluators`` (or equivalent) with ``run_case``.
        run_ledger: The session run ledger used for KPI reporting.
        case: The golden-anchor case dict.

    Returns:
        ``(eval_result, ai_output)`` so the caller can assert on the result.
    """
    case_id = case.get("case_id") or "unknown"

    user_input = case.get("user_input", "")
    result = ai_assistant.process_user_query(user_input)
    ai_output = extract_assistant_output(result)
    trace_logs = ai_assistant.trace_collector.get_trace_logs()

    # A case may declare simulated_output, the reply the assistant is assumed
    # to have produced (e.g. a hallucination-demo case fabricates a number so
    # the hallucination gate proves it is detected). Only this reply string is
    # overridden: trace_logs still come from the real agent run above, so the
    # hallucination gate compares the (simulated) reply's numbers against the
    # actually-fetched data.
    if case.get("simulated_output"):
        ai_output = case["simulated_output"]

    eval_result = evaluators_class.run_case(case, ai_output, trace_logs)

    run_ledger.record(
        case_id=case_id,
        case=case,
        eval_result=eval_result,
        ai_output=ai_output,
        metrics=eval_result.get("metrics"),
    )
    return eval_result, ai_output


def assert_golden_anchor(eval_result, ai_output, case_id="case"):
    """Assert a golden-anchor eval result passed (or skip on coverage gaps)."""
    assert isinstance(eval_result, dict), f"Result for {case_id} is not a dict"

    # If the evaluator reports missing coverage, skip the test to surface the gap.
    action = eval_result.get("action", "")
    if action == "MISSING_COVERAGE":
        pytest.skip(
            f"Skipping {case_id}: evaluator action={action}; "
            f"details={eval_result.get('details')}"
        )

    # Assert the test passed. Extract is_safe before the assert so pytest's
    # assertion rewriting has nothing to introspect (it would otherwise dump
    # the whole eval_result dict).
    is_safe = eval_result.get("is_safe") is True
    assert is_safe, (
        f"Golden anchor case {case_id} failed evaluation: {failure_summary(eval_result)}\n"
        f"{format_eval_result(eval_result)}\n"
        f"  AI output: {ai_output}"
    )
