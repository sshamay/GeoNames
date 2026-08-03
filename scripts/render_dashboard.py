#!/usr/bin/env python3
"""Render a self-contained AQuA KPI dashboard from reports/latest.json + history.jsonl.

Reads the AQuA run ledger written by tests/test_utils/aqua_reporting.py and
emits a single self-contained HTML file with:

  - headline KPI cards (pass rate, escape rate, confidence mean, sent to
    HITL, intent accuracy, hallucination rate)
  - the run timestamp, git commit and duration
  - pass-rate and confidence trend over all recorded runs
  - per-check pass-rate trend over all recorded runs, plus a sent-to-HITL
    trend (how many cases each run escalated)

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

  <div class="panel"><h2>Trend &mdash; pass rate &amp; aggregate confidence (all runs)</h2>
    <div class="chartbox">__TREND_SVG__</div>
  </div>

  <div class="panel"><h2>Per-check pass rate (all runs)</h2>
    <div class="chartbox">__PERCHECK_SVG__</div>
  </div>

  <div class="panel"><h2>Sent-to-HITL trend (all runs)</h2>
    <div class="chartbox">__HITL_SVG__</div>
  </div>
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
          <dd>Mean <code>aggregate_score</code> across the evaluated golden-anchor cases; each case&rsquo;s score is the average of its check scores.</dd>
          <dt>Sent to HITL</dt>
          <dd>Golden-anchor cases whose aggregate score fell below the threshold and were escalated to human-in-the-loop review (action <code>ESCALATE_TO_HITL</code>). They also count as failures.</dd>
          <dt>Per-check pass rate</dt>
          <dd>One line per quality gate (<code>content_rules</code>, <code>agent_logic</code>, <code>hallucination_check</code>, <code>expected_outcome</code>) across all runs: how often each gate passed. Runs before a gate existed show gaps.</dd>
          <dt>Sent-to-HITL trend</dt>
          <dd>How many golden-anchor cases were escalated to human-in-the-loop review (action <code>ESCALATE_TO_HITL</code>) in each run. Runs before this KPI existed show gaps.</dd>
          <dt>Trend charts</dt>
          <dd>One point per pytest session read from <code>history.jsonl</code>: pass rate and mean confidence over time. Runs before these KPIs existed show gaps.</dd>
          <dt>Legend</dt>
          <dd>Each line&rsquo;s label and color are shown in the legend above the chart, in the same order as the lines.</dd>
          <dt>Intent accuracy</dt>
          <dd>Share of golden-anchor cases whose executed API calls match the case&rsquo;s <code>required_tools</code> exactly &mdash; the called tool names form the same set, and every required tool that declares <code>parameters</code> was called with exactly those values. Cases that declare no tool requirement (e.g. the greeting fallback) are excluded from the denominator.</dd>
          <dt>Hallucination rate</dt>
          <dd>Share of cases where a number in the assistant&rsquo;s reply (earthquake count, strongest magnitude, weather-station count) does not match the raw JSON the fetchers returned, read from the trace&rsquo;s <code>tool_outputs</code>. Cases where nothing was fetched are excluded.</dd>
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
  const hitl = k.sent_to_hitl || {};
  const cards = [
    { label: "pass rate", value: fmtPct(k.pass_rate), cls: "green" },
    { label: "escape rate", value: fmtPct(k.escape_rate), cls: "red" },
    { label: "confidence mean", value: fmtNum(k.aggregate_confidence.mean), cls: "" },
    { label: "sent to hitl", value: hitl.count == null ? "n/a" : hitl.count, cls: hitl.count > 0 ? "amber" : "" },
    { label: "intent accuracy", value: fmtPct(intentAcc.rate), cls: "" },
    { label: "hallucination rate", value: fmtPct(halRate.rate), cls: halRate.rate > 0 ? "red" : "" },
  ];
  el("cards").innerHTML = cards.map(c =>
    '<div class="card"><div class="value ' + c.cls + '">' + c.value + '</div>' +
    '<div class="label">' + c.label + '</div></div>').join("");
})();

function fmtPct(v) { return v == null ? "n/a" : (v * 100).toFixed(1) + "%"; }
function fmtNum(v) { return v == null ? "n/a" : Number(v).toFixed(3); }
function fmtStamp(iso) {
  if (!iso) return "n/a";
  return new Date(iso).toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC");
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

def _axis_grid(pad_l, width, pad_r, pad_t, inner_h, vmax=1.0, value_format="pct"):
    parts = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = pad_t + inner_h * (1.0 - frac)
        label = f"{int(frac * vmax)}" if value_format == "int" else f"{int(frac * 100)}%"
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
            f'stroke="#334155" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{gy + 4:.1f}" fill="#94a3b8" font-size="10" '
            f'text-anchor="end">{label}</text>'
        )
    return parts


def line_chart_svg(
    series,
    labels,
    width=600,
    height=280,
    vmax=1.0,
    value_format="pct",
):
    """Multi-series line/area chart; y axis spans 0..vmax (pct or int labels)."""
    pad_l, pad_r, pad_t, pad_b = 44, 10, 36, 22
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    n = len(labels)

    def _x(i: int) -> float:
        return pad_l + inner_w * (i / max(n - 1, 1))

    def _y(v: float) -> float:
        return pad_t + inner_h * (1.0 - min(max(v, 0.0), vmax) / vmax)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
    ]
    parts.extend(_axis_grid(pad_l, width, pad_r, pad_t, inner_h, vmax, value_format))

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
        f'<rect x="{pad_l + i * 140}" y="12" width="10" height="10" fill="{s["color"]}"/>'
        f'<text x="{pad_l + i * 140 + 14}" y="21" fill="#e2e8f0" font-size="11">'
        f'{html.escape(s["label"])}</text>'
        for i, s in enumerate(series)
    )
    parts.append(legend)
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ data + render

_PER_CHECK_PALETTE = ["#4ade80", "#f87171", "#fbbf24", "#38bdf8", "#c084fc", "#fb7185", "#a3e635"]


def _build_data(report_dir: str) -> Dict[str, Any]:
    latest = _load_json(os.path.join(report_dir, "latest.json"))
    kpis = latest.get("kpis", {})

    history = _load_history(report_dir)
    labels = [_short_label(r.get("generated_at", "")) for r in history]

    trend = {
        "labels": labels,
        "passRate": [r.get("kpis", {}).get("pass_rate") for r in history],
        "confidence": [r.get("kpis", {}).get("aggregate_confidence", {}).get("mean")
                       for r in history],
        "hitl": [r.get("kpis", {}).get("sent_to_hitl", {}).get("count")
                 for r in history],
    }

    # Per-check series across all runs (one line per quality gate). Runs before
    # a gate existed carry None, which line_chart_svg renders as a gap.
    check_names = sorted({
        name for r in history
        for name in (r.get("kpis", {}).get("per_check") or {})
    })
    per_check_trend = {"passRate": []}
    for i, name in enumerate(check_names):
        color = _PER_CHECK_PALETTE[i % len(_PER_CHECK_PALETTE)]
        pass_vals = []
        for r in history:
            stat = (r.get("kpis", {}).get("per_check") or {}).get(name) or {}
            pass_vals.append(stat.get("pass_rate"))
        per_check_trend["passRate"].append({"label": name, "color": color, "values": pass_vals})

    return {
        "meta": {
            "generated_at": latest.get("generated_at"),
            "git_commit": latest.get("git_commit"),
            "duration_seconds": latest.get("duration_seconds"),
        },
        "kpis": kpis,
        "trend": trend,
        "perCheckTrend": per_check_trend,
    }


def render(report_dir: str, out_path: str) -> str:
    data = _build_data(report_dir)

    trend_svg = line_chart_svg(
        [
            {"label": "pass rate", "color": "#4ade80", "values": data["trend"]["passRate"]},
            {"label": "confidence", "color": "#fbbf24", "values": data["trend"]["confidence"]},
        ],
        data["trend"]["labels"],
    )
    percheck_svg = line_chart_svg(data["perCheckTrend"]["passRate"], data["trend"]["labels"])

    hitl_values = [v for v in data["trend"]["hitl"] if v is not None]
    hitl_max = max(hitl_values) if hitl_values else 1
    hitl_svg = line_chart_svg(
        [{"label": "sent to hitl", "color": "#fbbf24", "values": data["trend"]["hitl"]}],
        data["trend"]["labels"],
        vmax=hitl_max,
        value_format="int",
    )

    html_out = (
        TEMPLATE
        .replace("__TREND_SVG__", trend_svg)
        .replace("__PERCHECK_SVG__", percheck_svg)
        .replace("__HITL_SVG__", hitl_svg)
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
