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

pytestmark = pytest.mark.ai_assistant


def _extract_response(result):
    """
    Extract response from assistant return value.
    
    Phase 1: assistant returns (response, trace_logs) tuple
    Phase 2: assistant returns just response string
    
    This helper makes test compatible with both phases.
    """
    return result[0] if isinstance(result, tuple) else result


def _format_eval_result(eval_result):
    """
    Render an AQuA eval result as readable multi-line text so failing
    golden anchors show exactly which checks failed and why.
    """
    if not isinstance(eval_result, dict):
        return str(eval_result)

    score = eval_result.get("aggregate_score")
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)

    lines = [
        f"action: {eval_result.get('action')}",
        f"aggregate_score: {score_text}",
        f"is_safe: {eval_result.get('is_safe')}",
        "details:",
    ]
    for check in eval_result.get("details") or []:
        status = check.get("status")
        score = check.get("score")
        reason = check.get("reason") or ""
        line = f"  [{check.get('check_name')}] status={status} score={score}"
        if reason:
            line += f"  {reason}"
        lines.append(line)

    skipped = eval_result.get("skipped")
    if skipped is not None:
        lines.append(f"skipped: {skipped}")

    return "\n".join(lines)


def test_golden_anchor_case(ai_assistant, aqua_evaluators_class, run_ledger, case_and_id):
    """
    Generic golden anchor test: one pytest case per golden anchor entry.
    
    Requirements for fixtures:
    - ai_assistant: Must have process_user_query(input) and trace_collector.get_trace_logs()
    - aqua_evaluators_class: AQuAEvaluators class with run_case() method
    - run_ledger: Session-scoped AQuA run ledger (KPI reporting); records each
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
    ai_output = _extract_response(result)
    
    # Extract trace logs from assistant's trace_collector
    # Works identically in Phase 1 and Phase 2
    trace_logs = ai_assistant.trace_collector.get_trace_logs()
    
    # Run AQuA evaluators
    eval_result = aqua_evaluators_class.run_case(case, ai_output, trace_logs)
    
    # Record the run outcome for KPI reporting (before assertion so failures log too)
    run_ledger.record(case_id=case_id, case=case, eval_result=eval_result, ai_output=ai_output)
    
    assert isinstance(eval_result, dict), f"Result for {case_id} is not a dict"
    
    # If evaluator reports missing coverage, skip test to surface the gap
    action = eval_result.get("action", "")
    if action == "MISSING_COVERAGE":
        pytest.skip(
            f"Skipping {case_id}: evaluator action={action}; "
            f"details={eval_result.get('details')}"
        )
    
    # Assert test passed
    assert eval_result.get("is_safe") is True, (
        f"Golden anchor case {case_id} failed evaluation:\n"
        f"{_format_eval_result(eval_result)}\n"
        f"AI output: {ai_output}"
    )
