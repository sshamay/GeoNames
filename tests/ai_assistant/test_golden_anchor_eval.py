"""
Generic Golden Anchor Testing Framework.

This test module is REUSABLE across AI projects. It does not import
project-specific code directly. Instead, it uses fixtures provided by
conftest.py to get the assistant, evaluator, and test cases.

To use in another project:
1. Copy this file to your tests/ directory
2. Create conftest.py with fixtures:
   - @pytest.fixture ai_assistant(): return your_ai_class()
   - @pytest.fixture golden_anchor_cases(): return your_test_cases
   - @pytest.fixture golden_anchor_case_ids(): return case_ids
3. No other changes needed!
"""

import pytest

from test_utils.evaluation import (
    extract_assistant_output,
    failure_summary,
    format_eval_result,
)

pytestmark = pytest.mark.ai_assistant


def test_golden_anchor_case(ai_assistant, evaluator_class, run_ledger, case_and_id):
    """
    Generic golden anchor test: one pytest case per golden anchor entry.
    
    Requirements for fixtures:
    - ai_assistant: Must have process_user_query(input) and trace_collector.get_trace_logs()
    - evaluator_class: Evaluator class with run_case() method
    - run_ledger: Session-scoped evaluation run ledger (KPI reporting); records each
      case's eval result before assertions so failing cases are captured too.
    - case_and_id: Golden anchor case dict with case_id, user_input, etc.
    
    Design:
    - This test is generic and project-agnostic
    - All project-specific configuration is in conftest.py
    - Can be copied to other projects as-is
    """
    case = case_and_id
    case_id = case.get("case_id") or "unknown"
    
    # Call assistant (returns tuple in Phase 1, string in Phase 2)
    user_input = case.get("user_input", "")
    result = ai_assistant.process_user_query(user_input)
    
    # Extract response (works for both tuple and string)
    ai_output = extract_assistant_output(result)
    
    # Extract trace logs from assistant's trace_collector
    # Works identically in Phase 1 and Phase 2
    trace_logs = ai_assistant.trace_collector.get_trace_logs()
    
    # Generic harness feature: a case may declare simulated_output, the reply
    # the assistant is assumed to have produced (e.g. a hallucination-demo case
    # fabricates a number so the hallucination gate proves it is detected).
    if case.get("simulated_output"):
        ai_output = case["simulated_output"]
    
    # Run evaluators (includes the hallucination gate + KPI metrics)
    eval_result = evaluator_class.run_case(case, ai_output, trace_logs)
    
    # Record the run outcome for KPI reporting (before assertion so failing
    # cases are captured too). Per-case KPI metrics come from the evaluator.
    run_ledger.record(
        case_id=case_id,
        case=case,
        eval_result=eval_result,
        ai_output=ai_output,
        metrics=eval_result.get("metrics"),
    )
    
    assert isinstance(eval_result, dict), f"Result for {case_id} is not a dict"
    
    # If evaluator reports missing coverage, skip test to surface the gap
    action = eval_result.get("action", "")
    if action == "MISSING_COVERAGE":
        pytest.skip(
            f"Skipping {case_id}: evaluator action={action}; "
            f"details={eval_result.get('details')}"
        )
    
    # Assert test passed
    # Extract is_safe before the assert so pytest's assertion rewriting has
    # nothing to introspect (it would otherwise dump the whole eval_result dict).
    is_safe = eval_result.get("is_safe") is True
    assert is_safe, (
        f"Golden anchor case {case_id} failed evaluation: {failure_summary(eval_result)}\n"
        f"{format_eval_result(eval_result)}\n"
        f"  AI output: {ai_output}"
    )
