#!/usr/bin/env python3
"""Print an AI Quality Evaluation Report from the latest AQuA run.

Usage:
    python scripts/report.py
    python scripts/report.py --run reports/aqua_run_20260704_120000.json
"""

import json
import pathlib
import sys
from datetime import datetime, timezone


def _fmt_pct(value):
    """Format a ratio (0.0-1.0) as a percentage integer."""
    if value is None:
        return "N/A"
    return f"{int(round(value * 100))}%"


def _fmt_count(label, value):
    """Right-align a count value."""
    return f"{label}: {value}"


def render_report(report_path):
    """Read a JSON report and print the styled summary."""
    with open(report_path) as f:
        report = json.load(f)

    kpis = report.get("kpis", {})
    totals = kpis.get("totals", {})

    lines = []
    lines.append("AI Quality Evaluation Report")
    lines.append("=" * 40)
    lines.append("")

    # Counts
    total = totals.get("total", 0)
    passed = totals.get("passed", 0)
    failed = totals.get("failed", 0)

    lines.append(_fmt_count("Total Scenarios", total))
    lines.append("")

    # Quality dimensions
    intent = kpis.get("intent_accuracy", {})
    agent_logic = kpis.get("per_check", {}).get("agent_logic", {})
    rules = kpis.get("per_check", {}).get("content_rules", {})
    halluc = kpis.get("hallucination_rate", {})

    intent_rate = _fmt_pct(intent.get("rate"))
    tool_sel = _fmt_pct(agent_logic.get("pass_rate"))
    rules_rate = _fmt_pct(rules.get("pass_rate"))
    halluc_rate = _fmt_pct(halluc.get("rate"))

    lines.append(f"Intent Accuracy:        {intent_rate}")
    lines.append(f"Tool Selection Accuracy: {tool_sel}")
    lines.append(f"Business Rules:         {rules_rate}")
    lines.append(f"Hallucination Rate:     {halluc_rate}")
    lines.append("")

    # Overall quality score
    agg = kpis.get("aggregate_confidence", {})
    overall = _fmt_pct(agg.get("mean"))
    lines.append(f"Overall Quality Score:  {overall}")
    lines.append("")

    # Failed scenarios
    failed_cases = kpis.get("failed_cases", [])
    if failed_cases:
        lines.append("Failed Scenarios:")
        for fc in failed_cases:
            case_id = fc.get("case_id", "Unknown")
            checks = ", ".join(fc.get("failing_checks", []))
            lines.append(f"- {case_id}: {checks}")
    else:
        lines.append("Failed Scenarios:")
        lines.append("  (none)")

    lines.append("")
    return "\n".join(lines)


def main():
    report_path = None
    if len(sys.argv) > 1 and sys.argv[1] != "--run":
        report_path = sys.argv[1]
    elif "--run" in sys.argv:
        idx = sys.argv.index("--run")
        if idx + 1 < len(sys.argv):
            report_path = sys.argv[idx + 1]

    if report_path is None:
        latest = pathlib.Path("reports/latest.json")
        if latest.exists():
            report_path = str(latest)
        else:
            print("No report found. Run golden anchor tests first.", file=sys.stderr)
            sys.exit(1)

    print(render_report(report_path))


if __name__ == "__main__":
    main()
