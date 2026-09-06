"""HOST-PROJECT AQuA CONFIG - THE one place for all GeoNames aqua settings.

This is the GeoNames project's own file (it is NOT part of the shipped ``aqua``
framework package; that package ships only the generic template
``aqua/src/aqua/aqua_config.example.py``). The AQuA pytest plugin auto-loads
this file from the project root, so every knob below - judge, thresholds,
paths, and the hallucination definition - lives HERE and nowhere else.

Everything set here overrides the ``AQUA_*`` env vars, which remain only as
fallbacks. All values are written out explicitly - nothing is hidden behind a
default: if you see it, that is the value used.
"""

from __future__ import annotations

import re

from aqua.config import JudgeConfig, Thresholds

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
CASES_PATH = "tests/data/golden_anchor.json"   # 25 GeoNames cases (GN-001..025)
DATA_DIR = "tests/data"
REPORT_DIR = "reports"

# ---------------------------------------------------------------------------
# 2. Evaluation thresholds (ALL of them, with values)
# ---------------------------------------------------------------------------
THRESHOLDS = Thresholds(
    expected_outcome_semantic=0.6,  # P5 semantic similarity gate: below this, escalate to the judge
    llm_judge_pass=0.3,             # minimum P6 judge score (0..1) that counts as a pass
    default_case=0.9,               # aggregate confidence below which a case FAILS / escalates to HITL
)

# ---------------------------------------------------------------------------
# 3. LLM-as-a-judge
# ---------------------------------------------------------------------------
# Local Ollama judge: fully offline, fast, no data leaves the machine. Requires
# `ollama serve` running with the model pulled (`ollama pull qwen2.5:7b`).
# qwen2.5:7b is used over the lighter llama3.2:3b because grounding verdicts
# from 3b were unreliable (scored grounded, correct replies at 0.00).
# Each field maps 1:1 to an AQUA_JUDGE_* env var as a fallback.
JUDGE = JudgeConfig(
    enabled=True,          # set False to keep the suite offline/deterministic
    provider="ollama",     # openai_compatible | openai | aihorde | ollama
    model="qwen2.5:7b",    # required; no sane default model exists
    base_url=None,         # None = provider default (http://localhost:11434/v1)
    api_key=None,          # None = provider default ("ollama")
    timeout=60.0,
    rubric="groundedness_and_completeness",
    debug=True,            # attach raw judge output to verdicts (bloaty)
    system_prompt=None,    # None = framework default prompt (aqua.judge.DEFAULT_JUDGE_SYSTEM_PROMPT)
    max_tokens=256,        # cap on judge model output tokens
    max_attempts=3,        # retries on transport errors / unparseable verdicts
)

# ---------------------------------------------------------------------------
# 4. Hallucination definition (what counts as a hallucination)
# ---------------------------------------------------------------------------
# A case FAILS the hallucination_check gate when hallucination_extractor returns
# a non-empty list. Below is the full GeoNames definition, written out.
#
# GeoNames reply format (deterministic `summarize` in
# src/geonames/services/ask_location.py) makes number claims like:
#   "5 recent earthquakes", "strongest magnitude 4.6", "3 weather observations"
# Each claim is compared against the raw JSON the fetchers actually returned
# (from the trace's tool_outputs). A claim that contradicts the data is a
# hallucination. Empty list = consistent.

_EARTHQUAKES_COUNT = re.compile(r"(\d+) recent earthquakes")
_STRONGEST_MAGNITUDE = re.compile(r"strongest magnitude (\d+(?:\.\d+)?)")
_WEATHER_STATIONS = re.compile(r"weather observations from (\d+) stations")
_NO_EARTHQUAKES = re.compile(r"no recent earthquakes")
_NO_WEATHER = re.compile(r"no weather observations")


def _quake_claims(text: str, quakes: list, issues: list) -> None:
    if _NO_EARTHQUAKES.search(text):
        if quakes:
            issues.append(f"summary says 'no recent earthquakes' but raw JSON has {len(quakes)}")
    count = _EARTHQUAKES_COUNT.search(text)
    if count and int(count.group(1)) != len(quakes):
        issues.append(f"summary says {count.group(1)} earthquakes, raw JSON has {len(quakes)}")
    mag = _STRONGEST_MAGNITUDE.search(text)
    if mag:
        magnitudes = [q["magnitude"] for q in quakes]
        if magnitudes:
            expected = float(f"{max(magnitudes):.1f}")
            if float(mag.group(1)) != expected:
                issues.append(
                    f"summary says strongest magnitude {mag.group(1)}, raw JSON max is {expected}"
                )
        else:
            issues.append("summary reports a strongest magnitude but raw JSON has no earthquakes")


def _weather_claims(text: str, observations: list, issues: list) -> None:
    if _NO_WEATHER.search(text):
        if observations:
            issues.append(
                f"summary says 'no weather observations' but raw JSON has {len(observations)}"
            )
    count = _WEATHER_STATIONS.search(text)
    if count and int(count.group(1)) != len(observations):
        issues.append(
            f"summary says {count.group(1)} stations, raw JSON has {len(observations)}"
        )


def hallucination_extractor(ai_output: str, tool_outputs: dict) -> list:
    """List every number claim in the reply that contradicts the raw data.

    Empty list = the reply is consistent (or nothing was fetched to compare
    against). Each entry names the endpoint and the exact discrepancy, so the
    golden-anchor hallucination gate can fail a case with a precise reason.
    """
    if not tool_outputs:
        return []
    text = (ai_output or "").lower()
    issues: list = []
    for endpoint, data in tool_outputs.items():
        if endpoint == "earthquakes":
            _quake_claims(text, data.get("earthquakes") or [], issues)
        elif endpoint == "weather":
            _weather_claims(text, data.get("weatherObservations") or [], issues)
    return issues
