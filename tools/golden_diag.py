"""Golden-anchor failure diagnostic.

Runs every case in ``golden_anchor.json`` through the real agent + the real
AQuA evaluators (same wiring as ``tests/ai_assistant/test_golden_anchor_eval.py``)
and prints, per case, a classification of WHY it failed:

    JSON-SPEC   -> the golden-anchor JSON definition is wrong / needs your
                   input (e.g. a required keyword the agent is never expected
                   to say, a bbox param the agent can't produce, or an expected
                   outcome that contradicts live data).
    AGENT       -> the agent behaved wrongly (under-reported, fabricated,
                   dropped a tool, or failed). The JSON is fine; the agent
                   (or model) is the problem.
    ADVERSARIAL -> an adversarial / diagnostic case expected to fail.
    PASS        -> the case passed.

Usage:
    python -m tools.golden_diag [case_id ...]          # subset by case id
    python -m tools.golden_diag --all                  # all cases (default)
    python -m tools.golden_diag --json                 # JSON-lines output for scripting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aqua_config  # noqa: E402  (host project config: JUDGE, THRESHOLDS, extractor)
from aqua.evaluation import AQuAEvaluators  # noqa: E402
from aqua.judge import build_llm_judge  # noqa: E402

from geonames.config_loader import load_config  # noqa: E402
from geonames.factory import build_assistant  # noqa: E402


# --------------------------------------------------------------------------- #
# Classification helpers
# --------------------------------------------------------------------------- #


def _call_param_strs(calls):
    """Collapse executed tool calls into concise 'name(params)' strings."""
    out = []
    for c in calls:
        params = c.get("parameters") or {}
        pstr = ", ".join(f"{k}={v}" for k, v in params.items())
        out.append(f"{c.get('name')}({pstr})")
    return out


def _required_tools_are_bbox(case):
    """True when any required tool declares bbox params (legacy spec)."""
    for t in case.get("required_tools") or []:
        params = t.get("parameters") or {}
        if any(k in params for k in ("north", "south", "east", "west")):
            return True
    return False


def _classify(case, eval_result, ai_output, calls_str):
    """Return one of: PASS, ADVERSARIAL, JSON-SPEC, AGENT + a short reason."""
    details = {d.get("check_name"): d for d in eval_result.get("details") or []}

    # Scenario tag for deliberate-fail cases.
    scenario = (case.get("scenario") or "").lower()
    if "adversarial" in scenario or "must fail" in scenario:
        return "ADVERSARIAL", "adversarial/diagnostic case expected to fail by design"

    if eval_result.get("is_safe") is True:
        return "PASS", ""

    failed = {k: d.get("reason") for k, d in details.items() if d.get("status") == "FAILED"}

    # --- Agent clearly failed or returned an error ----------------------- #
    if ai_output and ("could not answer" in ai_output.lower()
                      or "request failed" in ai_output.lower()):
        return "AGENT", f"agent returned an error: {ai_output[:120]!r}"

    # --- Required tools not called ---------------------------------------- #
    if "agent_logic" in failed and "agent_logic" in details:
        if _required_tools_are_bbox(case):
            return ("JSON-SPEC",
                    "required_tools use bbox (north/south/east/west) params the "
                    "real agent never emits; the JSON must declare {'place': ...} "
                    "instead")
        # The agent's calls vs the required names echo a real routing bug.
        return ("AGENT",
                f"required tools not called; agent called: {calls_str or '<none>'}")

    # --- Keyword mismatch -------------------------------------------------- #
    if "content_rules" in failed:
        missing = _extract_missing_keywords(details["content_rules"].get("reason"))
        # Try a coarse semantic check: is the reply about the same topic?
        topic_ok = _loose_keywords_present(ai_output, case.get("required_keywords") or [])
        if topic_ok:
            return ("JSON-SPEC",
                    f"keyword list too rigid/out of sync: missing literal "
                    f"{missing}, but the reply covers the same topic. Either fix "
                    f"the keyword or the phrasing.")
        return ("AGENT", f"reply misses required keywords {missing} and does not "
                         f"even loosely cover them")

    # --- Hallucination / number mismatch ----------------------------------- #
    if "hallucination_check" in failed:
        return "AGENT", details["hallucination_check"].get("reason")

    # --- Expected-outcome insufficient ------------------------------------- #
    if "expected_outcome" in failed:
        return ("JSON-SPEC",
                details["expected_outcome"].get("reason")
                or "expected_outcome not matched; may contradict live data")

    # --- Anything else ----------------------------------------------------- #
    reason = "; ".join(f"{k}: {v}" for k, v in failed.items()) or "unknown"
    return "AGENT", reason


def _extract_missing_keywords(reason):
    if not reason:
        return []
    if "Missing required keywords:" in reason:
        return reason.split("Missing required keywords:")[1].strip()
    return reason


def _loose_keywords_present(ai_output, required_keywords):
    """Topic-level check: does any required keyword's words appear in output?"""
    text = (ai_output or "").lower()
    for kw in required_keywords or []:
        words = [w for w in kw.lower().split() if w not in ("near", "around", "the")]
        if words and not any(w in text for w in words):
            return False
    return True


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _wire_evaluators():
    AQuAEvaluators.llm_judge = build_llm_judge(aqua_config.JUDGE)
    ex = aqua_config.hallucination_extractor
    AQuAEvaluators.hallucination_extractor = staticmethod(ex) if ex is not None else None
    th = aqua_config.THRESHOLDS
    AQuAEvaluators.expected_outcome_semantic_threshold = th.expected_outcome_semantic
    AQuAEvaluators.llm_judge_pass_threshold = th.llm_judge_pass
    AQuAEvaluators.default_case_threshold = th.default_case


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_ids", nargs="*", metavar="GN-XXX",
                    help="only run these case ids (default: all)")
    ap.add_argument("--all", action="store_true", help="run all cases (default)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    cases_path = Path(ROOT) / aqua_config.CASES_PATH
    cases = json.loads(cases_path.read_text())
    if args.case_ids:
        want = set(args.case_ids)
        cases = [c for c in cases if c.get("case_id") in want]
    cases = sorted(cases, key=lambda c: c.get("case_id"))

    _wire_evaluators()
    settings = load_config(env="test")

    rows = []
    for idx, case in enumerate(cases, 1):
        cid = case.get("case_id")
        if not args.json:
            print(f"[{idx}/{len(cases)}] running {cid} ...", file=sys.stderr, flush=True)
        agent = build_assistant(settings)
        try:
            out = agent.process_user_query(case.get("user_input", ""))
        except Exception as exc:  # agent crashes -> surface as AGENT error
            out = f"EXCEPTION: {exc}"
        ai_output = out
        trace_logs = agent.trace_collector.get_trace_logs()
        # Simulated-output cases substitute the reply (same as golden_anchor.py).
        if case.get("simulated_output"):
            ai_output = case["simulated_output"]
        if args.json and case.get("simulated_output"):
            ai_output = case["simulated_output"]

        eval_result = AQuAEvaluators.run_case(case, ai_output, trace_logs)
        calls_str = ", ".join(_call_param_strs(trace_logs.get("executed_tool_calls") or []))
        verdict, reason = _classify(case, eval_result, ai_output, calls_str)

        rows.append({
            "case_id": cid,
            "scenario": case.get("scenario", ""),
            "verdict": verdict,
            "reason": reason,
            "is_safe": eval_result.get("is_safe"),
            "aggregate": eval_result.get("aggregate_score"),
            "ai_output": ai_output,
            "calls": calls_str,
        })

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    for r in rows:
        line = f"[{r['verdict']:10}] {r['case_id']}  {r['scenario']}"
        print(line)
        if r["verdict"] != "PASS":
            print(f"    reason: {r['reason']}")
        print(f"    calls : {r['calls'] or '<none>'}")
        print(f"    output: {r['ai_output'][:160]}")
        print()

    order = ["PASS", "JSON-SPEC", "AGENT", "ADVERSARIAL"]
    print("== SUMMARY ==")
    for k in order:
        print(f"  {k:10} {counts.get(k, 0)}")
    print(f"  {'TOTAL':10} {len(rows)}")


if __name__ == "__main__":
    main()
