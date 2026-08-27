#!/usr/bin/env python3
"""AQuA command-line interface.

Usage:
    aqua report                # print AI quality report from reports/latest.json
    aqua report --run <path>   # print report from a specific run JSON
    aqua dashboard             # render reports/dashboard.html
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

from aqua.dashboard import dashboard_main


def _fmt_pct(value):
    """Format a ratio (0.0-1.0) as a percentage integer."""
    if value is None:
        return "N/A"
    return f"{int(round(value * 100))}%"


def _fmt_count(label, value):
    """Right-align a count value."""
    return f"{label}: {value}"


def render_report(report_path):
    """Read a JSON report and return the styled summary text."""
    with open(report_path) as f:
        report = json.load(f)

    kpis = report.get("kpis", {})
    totals = kpis.get("totals", {})

    lines = []
    lines.append("AI Quality Evaluation Report")
    lines.append("=" * 40)
    lines.append("")

    total = totals.get("total", 0)
    lines.append(_fmt_count("Total Scenarios", total))
    lines.append("")

    intent = kpis.get("intent_accuracy", {})
    agent_logic = kpis.get("per_check", {}).get("agent_logic", {})
    rules = kpis.get("per_check", {}).get("content_rules", {})
    halluc = kpis.get("hallucination_rate", {})

    lines.append(f"Intent Accuracy:        {_fmt_pct(intent.get('rate'))}")
    lines.append(f"Tool Selection Accuracy: {_fmt_pct(agent_logic.get('pass_rate'))}")
    lines.append(f"Business Rules:         {_fmt_pct(rules.get('pass_rate'))}")
    lines.append(f"Hallucination Rate:     {_fmt_pct(halluc.get('rate'))}")
    lines.append("")

    agg = kpis.get("aggregate_confidence", {})
    lines.append(f"Overall Quality Score:  {_fmt_pct(agg.get('mean'))}")
    lines.append("")

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


def report_main() -> None:
    """Print the styled AI quality report (argparse-friendly)."""
    parser = argparse.ArgumentParser(description="Print the AI quality report.")
    parser.add_argument("--run", default=None, help="Path to a run JSON (default: reports/latest.json)")
    args = parser.parse_args()

    report_path = args.run
    if report_path is None:
        latest = pathlib.Path("reports/latest.json")
        if latest.exists():
            report_path = str(latest)
        else:
            print("No report found. Run golden anchor tests first.", file=sys.stderr)
            sys.exit(1)

    print(render_report(report_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="AQuA tooling", prog="aqua")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report", help="print the AI quality report from the latest run")
    sub.add_parser("dashboard", help="render reports/dashboard.html")
    args = parser.parse_args()

    # The sub-commands have their own argparsers; drop the command token so
    # their remaining arguments parse cleanly.
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if args.command == "dashboard":
        dashboard_main()
    else:
        report_main()
