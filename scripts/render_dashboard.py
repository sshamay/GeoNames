#!/usr/bin/env python3
"""Render a self-contained AQuA KPI dashboard from reports/latest.json + history.jsonl.

Reads the AQuA run ledger written by tests/test_utils/aqua_reporting.py and
emits a single self-contained HTML file with:

  - headline KPI cards (pass rate, escape rate, confidence, counts, sent to
    HITL, intent accuracy, hallucination rate, automation trust signal)
  - the run timestamp, git commit and duration
  - pass/escape-rate and confidence trend over all recorded runs, plus the
    automation trust signal trend
  - per-check pass rates and average scores for the latest run
  - failed cases with reasons, coverage gaps, and the pytest suite summary

Charts are inline SVG generated at build time (no CDN, no JS libraries), so the
dashboard renders offline in any browser.

Usage:
    python scripts/render_dashboard.py [--report-dir reports] [--out reports/dashboard.html]

Stdlib only.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from typing import Any, Dict, List, Optional

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AQuA Golden Anchor Dashboard</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --line:#334155; --text:#e2e8f0;
          --muted:#94a3b8; --green:#4ade80; --red:#f87171; --amber:#fbbf24; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background:var(--bg); color:var(--text); }
  header { padding:20px 28px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:20px; }
  header .meta { color:var(--muted); font-size:13px; margin-top:4px; }
  main { padding:20px 28px; max-width:1200px; margin:0 auto; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:14px; margin-bottom:22px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  .card .value { font-size:26px; font-weight:700; }
  .card .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  .green { color:var(--green); } .red { color:var(--red); } .amber { color:var(--amber); }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  @media (max-width:900px) { .grid2 { grid-template-columns:1fr; } }
  .layout { display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:18px; align-items:start; }
  .sidebar { position:sticky; top:16px; }
  @media (max-width:1100px) { .layout { grid-template-columns:1fr; } .sidebar { position:static; } }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:18px; }
  .panel h2 { margin:0 0 12px; font-size:15px; color:var(--text); }
  .sidebar dl { margin:0; }
  .sidebar dt { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:#7dd3fc; margin:14px 0 3px; }
  .sidebar dt:first-child { margin-top:0; }
  .sidebar dd { margin:0 0 2px; font-size:12px; line-height:1.45; color:var(--muted); }
  .chartbox { position:relative; width:100%; }
  .chartbox svg { display:block; max-width:100%; height:auto; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
  .pill { display:inline-block; padding:2px 8px; border-radius:99px; font-size:11px; font-weight:600; }
  .pill.fail { background:rgba(248,113,113,.15); color:var(--red); }
  .pill.pass { background:rgba(74,222,128,.15); color:var(--green); }
  code { background:rgba(148,163,184,.12); padding:1px 5px; border-radius:4px; font-size:12px; }
  .muted { color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>AQuA Golden Anchor Dashboard</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <div class="layout">
    <div class="content">
  <div class="cards" id="cards"></div>

  <div class="panel"><h2>Trend &mdash; pass / escape rate &amp; aggregate confidence (all runs)</h2>
    <div class="chartbox">__TREND_SVG__</div>
  </div>

  <div class="panel"><h2>Automation trust signal (all runs)</h2>
    <div class="chartbox">__TRUST_SVG__</div>
  </div>

  <div class="grid2">
    <div class="panel"><h2>Per-check pass rate (latest run)</h2>
      <div class="chartbox">__PERCHECK_SVG__</div>
    </div>
    <div class="panel"><h2>Per-check average score (latest run)</h2>
      <div class="chartbox">__PERCHECK_SCORE_SVG__</div>
    </div>
  </div>

  <div class="grid2">
    <div class="panel" id="failedPanel"><h2>Failed cases</h2></div>
    <div class="panel" id="coveragePanel"><h2>Coverage gaps</h2></div>
  </div>

  <div class="panel" id="pytestPanel"><h2>Pytest suite</h2></div>
    </div>

    <aside class="sidebar">
      <div class="panel">
        <h2>How to read this dashboard</h2>
        <dl>
          <dt>Run</dt>
          <dd>Timestamp of the latest pytest run this dashboard was generated from (UTC), plus the git commit and session duration.</dd>
          <dt>Pass rate</dt>
          <dd>Share of evaluated golden-anchor cases that passed: <code>passed &divide; (passed + failed)</code>.</dd>
          <dt>Escape rate</dt>
          <dd>Share that failed &mdash; defects the quality gate caught: <code>failed &divide; (passed + failed)</code>.</dd>
          <dt>Confidence</dt>
          <dd>Mean / min / max of each case&rsquo;s <code>aggregate_score</code>, which is the average of that case&rsquo;s check scores.</dd>
          <dt>Evaluated / passed / failed</dt>
          <dd>Counts from the latest run. Coverage gaps and no-tests-run cases are excluded from the pass rate on purpose.</dd>
          <dt>Sent to HITL</dt>
          <dd>Golden-anchor cases whose aggregate score fell below the threshold and were escalated to human-in-the-loop review (action <code>ESCALATE_TO_HITL</code>). They also count as failures.</dd>
          <dt>Per-check pass rate / avg score</dt>
          <dd>Per quality gate (<code>content_rules</code>, <code>agent_logic</code>, <code>hallucination_check</code>, <code>expected_outcome</code>): how often each passed and its mean score across all cases in the latest run.</dd>
          <dt>Trend charts</dt>
          <dd>One point per pytest session read from <code>history.jsonl</code>: pass/escape rate and mean confidence over time, plus the deterministic share of the suite (the automation trust signal). Runs before these KPIs existed show gaps.</dd>
          <dt>Failed cases</dt>
          <dd>Cases where <code>is_safe = False</code>, listing the failing checks and the evaluator&rsquo;s reason (e.g. missing keyword, semantic similarity below 0.85, or reply numbers contradicting fetched data).</dd>
          <dt>Coverage gaps</dt>
          <dd>Cases with <code>MISSING_COVERAGE</code> &mdash; a check was skipped (e.g. no <code>expected_outcome</code> declared), so the case cannot count as a pass.</dd>
          <dt>Pytest suite</dt>
          <dd>Raw pytest outcomes across the whole session, including the exact failed node ids.</dd>
          <dt>Intent accuracy</dt>
          <dd>Share of golden-anchor cases whose executed API calls exactly match the case&rsquo;s <code>required_tools</code>. Cases that declare no tool requirement (e.g. the greeting fallback) are excluded from the denominator.</dd>
          <dt>Hallucination rate</dt>
          <dd>Share of cases where a number in the assistant&rsquo;s reply (earthquake count, strongest magnitude, weather-station count) does not match the raw JSON the fetchers returned, read from the trace&rsquo;s <code>tool_outputs</code>. Cases where nothing was fetched are excluded.</dd>
          <dt>Automation trust signal</dt>
          <dd>Share of the suite classified as deterministic (offline <code>unit</code> tests) vs probabilistic (live API, user flows, golden anchors). Higher means more of the suite is reproducible offline.</dd>
        </dl>
      </div>
    </aside>
  </div>
</main>

<script>
const DATA = __AQUA_DATA__;

function el(id) { return document.getElementById(id); }

(function () {
  const meta = DATA.meta;
  el("meta").textContent =
    "Run " + fmtStamp(meta.generated_at) + " \\u00b7 git " + (meta.git_commit || "n/a") +
    " \\u00b7 duration " + meta.duration_seconds + "s";
  const k = DATA.kpis;
  const intentAcc = k.intent_accuracy || {};
  const halRate = k.hallucination_rate || {};
  const trust = k.automation_trust_signal || {};
  const hitl = k.sent_to_hitl || {};
  const cards = [
    { label: "pass rate", value: fmtPct(k.pass_rate), cls: "green" },
    { label: "escape rate", value: fmtPct(k.escape_rate), cls: "red" },
    { label: "confidence mean", value: fmtNum(k.aggregate_confidence.mean), cls: "" },
    { label: "confidence min", value: fmtNum(k.aggregate_confidence.min), cls: "" },
    { label: "confidence max", value: fmtNum(k.aggregate_confidence.max), cls: "" },
    { label: "evaluated", value: k.totals.evaluated, cls: "" },
    { label: "passed", value: k.totals.passed, cls: "green" },
    { label: "failed", value: k.totals.failed, cls: "red" },
    { label: "sent to hitl", value: hitl.count == null ? "n/a" : hitl.count, cls: hitl.count > 0 ? "amber" : "" },
    { label: "coverage gaps", value: k.totals.coverage_gaps, cls: k.totals.coverage_gaps ? "amber" : "" },
    { label: "intent accuracy", value: fmtPct(intentAcc.rate), cls: "" },
    { label: "hallucination rate", value: fmtPct(halRate.rate), cls: halRate.rate > 0 ? "red" : "" },
    { label: "automation trust", value: fmtPct(trust.rate), cls: "green" },
  ];
  el("cards").innerHTML = cards.map(c =>
    '<div class="card"><div class="value ' + c.cls + '">' + c.value + '</div>' +
    '<div class="label">' + c.label + '</div></div>').join("");
})();

function rows(items, cols) {
  if (!items.length) return '<p class="muted">None</p>';
  const head = "<tr>" + cols.map(c => "<th>" + c.label + "</th>").join("") + "</tr>";
  const body = items.map(it => "<tr>" + cols.map(c => "<td>" + c.render(it) + "</td>").join("") + "</tr>").join("");
  return "<table>" + head + body + "</table>";
}

el("failedPanel").innerHTML =
  "<h2>Failed cases</h2>" +
  rows(DATA.failedCases, [
    { label: "case", render: c => '<code>' + c.case_id + "</code>" },
    { label: "score", render: c => fmtNum(c.aggregate_score) },
    { label: "failing checks", render: c => (c.failing_checks || []).map(ch =>
        '<span class="pill fail">' + ch + "</span>").join(" ") || "\\u2014" },
    { label: "reason", render: c => esc(c.reason) },
  ]);

el("coveragePanel").innerHTML =
  "<h2>Coverage gaps</h2>" +
  rows(DATA.coverageGaps, [
    { label: "case", render: c => '<code>' + c.case_id + "</code>" },
    { label: "skipped checks", render: c => (c.skipped_checks || []).join(", ") || "\\u2014" },
  ]);

const ps = DATA.pytest;
el("pytestPanel").innerHTML =
  "<h2>Pytest suite</h2>" +
  "<p>passed <span class='pill pass'>" + ps.passed + "</span> \\u00b7 " +
  "failed <span class='pill fail'>" + ps.failed + "</span> \\u00b7 " +
  "skipped " + ps.skipped + " \\u00b7 errors " + ps.errors + "</p>" +
  rows((ps.failed_nodeids || []).map(n => ({ n })), [
    { label: "failed node ids", render: c => "<code>" + esc(c.n) + "</code>" },
  ]);

function fmtPct(v) { return v == null ? "n/a" : (v * 100).toFixed(1) + "%"; }
function fmtNum(v) { return v == null ? "n/a" : Number(v).toFixed(3); }
function fmtStamp(iso) {
  if (!iso) return "n/a";
  return new Date(iso).toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC");
}
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}
</script>
</body>
</html>
"""


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_history(report_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(report_dir, "history.jsonl")
    if not os.path.exists(path):
        return []
    runs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def _short_label(iso: str) -> str:
    return iso[11:19] if len(iso) >= 19 else iso


# ------------------------------------------------------------------- SVG charts

def _axis_grid(pad_l: int, width: int, pad_r: int, pad_t: int, inner_h: int) -> List[str]:
    parts = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = pad_t + inner_h * (1.0 - frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
            f'stroke="#334155" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" fill="#94a3b8" font-size="10" '
            f'text-anchor="end">{int(frac * 100)}%</text>'
        )
    return parts


def line_chart_svg(
    series: List[Dict[str, Any]],
    labels: List[str],
    width: int = 600,
    height: int = 280,
) -> str:
    """Multi-series line/area chart with value-scaled y axis (0..1)."""
    pad_l, pad_r, pad_t, pad_b = 44, 10, 14, 30
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(labels)

    def _x(i: int) -> float:
        return pad_l + inner_w * (i / max(n - 1, 1))

    def _y(v: float) -> float:
        return pad_t + inner_h * (1.0 - min(max(v, 0.0), 1.0))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
    ]
    parts.extend(_axis_grid(pad_l, width, pad_r, pad_t, inner_h))

    step = max(n // 10, 1)
    for i in range(0, n, step):
        parts.append(
            f'<text x="{_x(i):.1f}" y="{height - 10}" fill="#94a3b8" font-size="9" '
            f'text-anchor="middle">{html.escape(labels[i])}</text>'
        )

    for s in series:
        points = []
        prev: Optional[float] = None
        for i, v in enumerate(s["values"]):
            if v is None:
                v = prev
            else:
                prev = v
            if v is None:
                continue
            points.append(f"{_x(i):.1f},{_y(v):.1f}")
        if points:
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{s["color"]}" stroke-width="2"/>'
            )

    legend = "".join(
        f'<rect x="{pad_l + i * 130}" y="{height - 22}" width="10" height="10" fill="{s["color"]}"/>'
        f'<text x="{pad_l + i * 130 + 14}" y="{height - 13}" fill="#e2e8f0" font-size="11">'
        f'{html.escape(s["label"])}</text>'
        for i, s in enumerate(series)
    )
    parts.append(legend)
    parts.append("</svg>")
    return "".join(parts)


def bar_chart_svg(
    names: List[str],
    values: List[Optional[float]],
    color: str,
    width: int = 300,
    height: int = 280,
) -> str:
    """Vertical bar chart with value labels (values are 0..1)."""
    pad_l, pad_r, pad_t, pad_b = 44, 8, 22, 46
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(names)
    slot = inner_w / max(n, 1)
    bar_w = min(slot * 0.55, 70)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    parts.extend(_axis_grid(pad_l, width, pad_r, pad_t, inner_h))

    for i, (name, v) in enumerate(zip(names, values)):
        vv = min(max(v if v is not None else 0.0, 0.0), 1.0)
        bx = pad_l + slot * i + (slot - bar_w) / 2
        bh = vv * inner_h
        cx = pad_l + slot * i + slot / 2
        parts.append(
            f'<rect x="{bx:.1f}" y="{pad_t + inner_h - bh:.1f}" width="{bar_w:.1f}" '
            f'height="{max(bh, 1):.1f}" rx="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{pad_t + inner_h - bh - 6:.1f}" fill="#e2e8f0" '
            f'font-size="10" text-anchor="middle">{vv:.0%}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{height - 12}" fill="#94a3b8" font-size="10" '
            f'text-anchor="middle">{html.escape(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ data + render

def _build_data(report_dir: str) -> Dict[str, Any]:
    latest = _load_json(os.path.join(report_dir, "latest.json"))
    kpis = latest.get("kpis", {})
    per_check = kpis.get("per_check", {})

    history = _load_history(report_dir)
    trend = {
        "labels": [_short_label(r.get("generated_at", "")) for r in history],
        "passRate": [r.get("kpis", {}).get("pass_rate") for r in history],
        "escapeRate": [r.get("kpis", {}).get("escape_rate") for r in history],
        "confidence": [r.get("kpis", {}).get("aggregate_confidence", {}).get("mean")
                       for r in history],
        "trust": [r.get("kpis", {}).get("automation_trust_signal", {}).get("rate")
                  for r in history],
    }

    return {
        "meta": {
            "generated_at": latest.get("generated_at"),
            "git_commit": latest.get("git_commit"),
            "duration_seconds": latest.get("duration_seconds"),
        },
        "kpis": kpis,
        "trend": trend,
        "perCheck": {
            "names": sorted(per_check),
            "passRate": [per_check[n].get("pass_rate") for n in sorted(per_check)],
            "avgScore": [per_check[n].get("avg_score") for n in sorted(per_check)],
        },
        "failedCases": kpis.get("failed_cases", []),
        "coverageGaps": kpis.get("coverage_gap_cases", []),
        "pytest": latest.get("pytest_suite", {}),
    }


def render(report_dir: str, out_path: str) -> str:
    data = _build_data(report_dir)

    trend_svg = line_chart_svg(
        [
            {"label": "pass rate", "color": "#4ade80", "values": data["trend"]["passRate"]},
            {"label": "escape rate", "color": "#f87171", "values": data["trend"]["escapeRate"]},
            {"label": "confidence", "color": "#fbbf24", "values": data["trend"]["confidence"]},
        ],
        data["trend"]["labels"],
    )
    trust_svg = line_chart_svg(
        [
            {"label": "deterministic share", "color": "#c084fc", "values": data["trend"]["trust"]},
        ],
        data["trend"]["labels"],
    )
    percheck_svg = bar_chart_svg(
        data["perCheck"]["names"], data["perCheck"]["passRate"], "#4ade80"
    )
    percheck_score_svg = bar_chart_svg(
        data["perCheck"]["names"], data["perCheck"]["avgScore"], "#38bdf8"
    )

    html_out = (
        TEMPLATE
        .replace("__TREND_SVG__", trend_svg)
        .replace("__TRUST_SVG__", trust_svg)
        .replace("__PERCHECK_SVG__", percheck_svg)
        .replace("__PERCHECK_SCORE_SVG__", percheck_score_svg)
        .replace("__AQUA_DATA__", json.dumps(data))
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html_out)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--out", default=os.path.join("reports", "dashboard.html"))
    args = parser.parse_args()
    out = render(args.report_dir, args.out)
    print(f"Dashboard written to {out}")


if __name__ == "__main__":
    main()
