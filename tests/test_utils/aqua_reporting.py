"""
AQuA Run KPIs: measure and log golden anchor run success.

Collects one entry per golden anchor case evaluated during a pytest session,
computes reporting KPIs (pass rate, escape rate, aggregate confidence,
per-check pass rates, coverage gaps) and writes:

    reports/aqua_run_<timestamp>.json   full run detail (incl. per-case entries)
    reports/latest.json                 same content, stable name for dashboards
    reports/history.jsonl               one summary line appended per run (trend)

GENERIC: no project imports. Configure the output directory with the
AQUA_REPORT_DIR env var (default "reports", relative to the run directory).
"""

import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone

REPORT_DIR = os.environ.get("AQUA_REPORT_DIR", "reports")

# AQuAEvaluators.run_case() "action" -> KPI bucket.
_STATUS_BY_ACTION = {
    "RELEASE": "PASSED",
    "ESCALATE_TO_HITL": "FAILED",
    "MISSING_COVERAGE": "COVERAGE_GAP",
    "NO_TESTS_RUN": "NO_TESTS_RUN",
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class AQuARunLedger:
    """Records per-case golden anchor results and pytest outcomes across a run."""

    def __init__(self, report_dir=None):
        self.entries = []
        self.pytest_counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        self.pytest_failed_nodeids = []
        self.deterministic_count = 0
        self.probabilistic_count = 0
        self.started_at = _utc_now()
        self.started_epoch = time.time()
        self.duration_seconds = 0.0
        self.report_dir = report_dir or REPORT_DIR
        self.git_commit = _git_commit()

    # ------------------------------------------------------------------ record
    def record(self, case_id, case, eval_result, ai_output=None, metrics=None):
        """Store one evaluated case. Returns the normalized entry dict.

        ``metrics`` is an optional dict of bool-or-None project KPIs per case
        (e.g. ``{"intent_accurate": True, "hallucinated": False}``). None means
        the metric was not applicable and is excluded from its rate.
        """
        checks = eval_result.get("details") or []
        entry = {
            "case_id": case_id,
            "scenario": case.get("scenario"),
            "user_input": case.get("user_input"),
            "status": _STATUS_BY_ACTION.get(eval_result.get("action"), "NO_TESTS_RUN"),
            "action": eval_result.get("action"),
            "is_safe": eval_result.get("is_safe"),
            "aggregate_score": eval_result.get("aggregate_score"),
            "skipped_check_count": eval_result.get("skipped", 0),
            "failing_checks": [c["check_name"] for c in checks if c.get("status") == "FAILED"],
            "skipped_checks": [c["check_name"] for c in checks if c.get("status") == "SKIPPED"],
            "checks": checks,
            "metrics": dict(metrics or {}),
        }
        if ai_output is not None:
            entry["ai_output"] = ai_output
        self.entries.append(entry)
        return entry

    def note_pytest_outcome(self, nodeid, outcome):
        """Track the raw pytest result for every test (whole-suite context)."""
        if outcome in self.pytest_counts:
            self.pytest_counts[outcome] += 1
        if outcome == "failed":
            self.pytest_failed_nodeids.append(nodeid)

    def note_determinism(self, is_deterministic):
        """Classify one test as deterministic (offline) or probabilistic (live)."""
        if is_deterministic:
            self.deterministic_count += 1
        else:
            self.probabilistic_count += 1

    # ------------------------------------------------------------- KPI math
    def build_kpis(self):
        """Aggregate recorded entries into KPI summary dicts (no I/O)."""
        totals = {"total": len(self.entries), "passed": 0, "failed": 0,
                  "coverage_gaps": 0, "no_tests_run": 0}
        scores = []
        failed_cases = []
        coverage_gap_cases = []
        per_check = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0,
                                         "skipped": 0, "score_sum": 0.0})

        status_key = {"PASSED": "passed", "FAILED": "failed",
                      "COVERAGE_GAP": "coverage_gaps", "NO_TESTS_RUN": "no_tests_run"}
        for entry in self.entries:
            key = status_key.get(entry["status"])
            if key:
                totals[key] += 1
            if entry["aggregate_score"] is not None:
                scores.append(entry["aggregate_score"])
            if entry["status"] == "FAILED":
                reasons = "; ".join(
                    c.get("reason", "") for c in entry["checks"] if c.get("status") == "FAILED")
                failed_cases.append({
                    "case_id": entry["case_id"],
                    "aggregate_score": entry["aggregate_score"],
                    "failing_checks": entry["failing_checks"],
                    "reason": reasons,
                })
            elif entry["status"] == "COVERAGE_GAP":
                coverage_gap_cases.append({
                    "case_id": entry["case_id"],
                    "skipped_checks": entry["skipped_checks"],
                })
            for check in entry["checks"]:
                name = check["check_name"]
                stat = per_check[name]
                stat["total"] += 1
                cstatus = check.get("status")
                if cstatus == "PASSED":
                    stat["passed"] += 1
                    stat["score_sum"] += check.get("score") or 0.0
                elif cstatus == "FAILED":
                    stat["failed"] += 1
                    stat["score_sum"] += check.get("score") or 0.0
                else:
                    stat["skipped"] += 1

        totals["evaluated"] = totals["passed"] + totals["failed"]
        evaluated = totals["evaluated"]
        hitl_count = sum(1 for e in self.entries if e.get("action") == "ESCALATE_TO_HITL")
        judge_count = sum(
            1 for e in self.entries
            if any(c.get("check_name") == "llm_judge" for c in e.get("checks") or [])
        )
        totals["hitl"] = hitl_count
        totals["sent_to_judge"] = judge_count

        # Bool-or-None project KPIs recorded per case (None = not applicable).
        metric_rates = {}
        for name in ("intent_accurate", "hallucinated"):
            applicable = [e for e in self.entries if e.get("metrics", {}).get(name) is not None]
            positive = [e for e in applicable if e["metrics"][name] is True]
            metric_rates[name] = {
                "evaluated": len(applicable),
                "positive": len(positive),
                "negative": len(applicable) - len(positive),
                "rate": (len(positive) / len(applicable)) if applicable else None,
            }

        trust_total = self.deterministic_count + self.probabilistic_count

        return {
            "totals": totals,
            "pass_rate": totals["passed"] / evaluated if evaluated else None,
            "escape_rate": totals["failed"] / evaluated if evaluated else None,
            "intent_accuracy": metric_rates["intent_accurate"],
            "hallucination_rate": metric_rates["hallucinated"],
            "sent_to_hitl": {
                "count": hitl_count,
                "rate": (hitl_count / evaluated) if evaluated else None,
            },
            "sent_to_judge": {
                "count": judge_count,
                "rate": (judge_count / evaluated) if evaluated else None,
            },
            "automation_trust_signal": {
                "deterministic": self.deterministic_count,
                "probabilistic": self.probabilistic_count,
                "total": trust_total,
                "rate": (self.deterministic_count / trust_total) if trust_total else None,
            },
            "aggregate_confidence": {
                "mean": (sum(scores) / len(scores)) if scores else None,
                "min": min(scores) if scores else None,
                "max": max(scores) if scores else None,
                "count": len(scores),
            },
            "failed_cases": failed_cases,
            "coverage_gap_cases": coverage_gap_cases,
            "per_check": {
                name: {
                    "total": stat["total"],
                    "passed": stat["passed"],
                    "failed": stat["failed"],
                    "skipped": stat["skipped"],
                    "pass_rate": stat["passed"] / (stat["passed"] + stat["failed"])
                    if (stat["passed"] + stat["failed"]) else None,
                    "avg_score": stat["score_sum"] / stat["total"] if stat["total"] else None,
                }
                for name, stat in sorted(per_check.items())
            },
        }

    # ------------------------------------------------------------- reporting
    def build_report(self, mutation_id=None):
        """Full report dict: metadata + golden anchor KPIs + pytest context."""
        return {
            "report_type": "aqua_golden_anchor_kpis",
            "generated_at": _utc_now(),
            "git_commit": self.git_commit,
            "mutation_id": mutation_id,
            "duration_seconds": round(self.duration_seconds, 3),
            "kpis": self.build_kpis(),
            "pytest_suite": {
                **self.pytest_counts,
                "deterministic_tests": self.deterministic_count,
                "probabilistic_tests": self.probabilistic_count,
                "failed_nodeids": self.pytest_failed_nodeids,
            },
            "entries": self.entries,
        }

    def write_reports(self, mutation_id=None):
        """Write run JSON, latest.json and append the history trend line."""
        os.makedirs(self.report_dir, exist_ok=True)
        report = self.build_report(mutation_id)
        run_stamp = time.strftime("%Y%m%d_%H%M%S")

        run_path = os.path.join(self.report_dir, f"aqua_run_{run_stamp}.json")
        latest_path = os.path.join(self.report_dir, "latest.json")
        with open(run_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        latest = {k: v for k, v in report.items() if k != "entries"}
        with open(latest_path, "w", encoding="utf-8") as fh:
            json.dump(latest, fh, indent=2)

        history_path = os.path.join(self.report_dir, "history.jsonl")
        trend = {k: v for k, v in latest.items() if k != "entries"}
        trend["run_file"] = os.path.basename(run_path)
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(trend, sort_keys=True) + "\n")

        return {"run": run_path, "latest": latest_path, "history": history_path}
