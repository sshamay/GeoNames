"""Project-specific claims extractor for the AQuA hallucination gate.

Registers with the generic AQuA evaluator as ``hallucination_extractor`` (see
tests/conftest.py). Given the assistant's reply and the raw JSON the fetchers
returned (from the trace's ``tool_outputs``), it lists every number in the
reply that contradicts the data (earthquake count, strongest magnitude,
weather-station count). Empty list = consistent.

PROJECT-SPECIFIC: understands the deterministic ``summarize`` output format in
src/geonames/services/ask_location.py. Keep it in sync with that format.
"""

from __future__ import annotations

import re
from typing import Any, Dict

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
            # summarize() renders with {:.1f} (round-half-even), so compare the
            # raw max the same way instead of expecting an exact float match.
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


def claim_mismatches(ai_output: str, tool_outputs: Dict[str, Any]) -> list:
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
