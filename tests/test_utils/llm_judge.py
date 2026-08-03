"""LLM-as-a-Judge client for the golden-anchor evaluation.

Implements the ``Evaluator.llm_judge`` hook signature::

    callable(ai_output, expected_outcome, retrieved_context)
        -> {"score": float (0..1), "reason": str}

Talks to any OpenAI-compatible ``/chat/completions`` endpoint. Build the hook
from settings with :func:`build_llm_judge`, which returns ``None`` when the
feature is disabled or underconfigured, so the suite stays fully offline and
deterministic by default.

Works with keyless providers too: AI Horde's anonymous access uses the literal
API key ``0000000000`` and exposes OpenAI-compatible endpoints, so nothing
about the wire format changes.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from geonames.config_loader import Settings

JudgeFn = Callable[[str, str, List[Any]], Dict[str, Any]]

_JUDGE_MAX_TOKENS = 256
_JUDGE_MAX_ATTEMPTS = 3
_UNPARSEABLE_PREFIX = "Judge returned unparseable output:"

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict quality judge for an AI assistant that answers questions "
    "about a location (recent earthquakes, strongest magnitude, weather station "
    "observations). Score the assistant's reply for groundedness in the "
    "retrieved data and completeness against the golden reference answer. "
    "Output exactly one JSON object and nothing else - no prose, no markdown "
    "fences: {\"score\": 0.0-1.0, \"reason\": \"short rationale\"}."
)


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the first ``{...}`` JSON object in ``text``, or None.

    Tolerates models that wrap the verdict JSON in prose or markdown. Fails
    closed (returns None) when no parseable JSON object is present.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    start = text.find("{")
    while start != -1:
        end = text.find("}", start + 1)
        if end == -1:
            break
        candidate = text[start:end + 1]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def _extract_verdict(content: str) -> Dict[str, Any]:
    """Parse ``{"score": ..., "reason": ...}`` from model output.

    Tolerates markdown-fenced and prose-wrapped JSON. Returns
    ``{"score": 0.0, ...}`` when the output cannot be parsed so an
    uncooperative judge fails closed.
    """
    text = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    payload = _parse_json_object(text)
    if payload is not None:
        try:
            return {"score": float(payload["score"]), "reason": str(payload.get("reason", ""))}
        except (KeyError, TypeError, ValueError):
            pass
    return {"score": 0.0, "reason": _UNPARSEABLE_PREFIX + " " + text[:200]}


class _OpenAIJudge:
    """Minimal OpenAI-compatible chat-completions client used as the judge."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float, rubric: str):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._rubric = rubric

    def __call__(self, ai_output: str, expected_outcome: str, retrieved_context: List[Any]) -> Dict[str, Any]:
        user_prompt = (
            f"Rubric: {self._rubric}\n"
            f"Golden reference answer:\n{expected_outcome}\n\n"
            f"Retrieved context (raw fetched data):\n{json.dumps(retrieved_context, indent=1)}\n\n"
            f"Assistant reply:\n{ai_output}\n\n"
            "How well is the reply grounded in the retrieved context and does it "
            "satisfy the golden reference? Return strict JSON as instructed."
        )
        # Community/volunteer endpoints (e.g. AI Horde anonymous) occasionally
        # return degenerate or truncated replies from a bad worker, so retry a
        # couple of times on transport errors and unparseable verdicts - the next
        # attempt usually lands on a different worker.
        last_error = None
        for attempt in range(_JUDGE_MAX_ATTEMPTS):
            try:
                response = requests.post(
                    self._base_url + "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "temperature": 0,
                        "max_tokens": _JUDGE_MAX_TOKENS,
                        "messages": [
                            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=self._timeout,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                last_error = f"LLM judge call failed: {exc}"
                if attempt < _JUDGE_MAX_ATTEMPTS - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return {"score": 0.0, "reason": last_error}

            verdict = _extract_verdict(content)
            if not verdict["reason"].startswith(_UNPARSEABLE_PREFIX):
                if not isinstance(verdict["score"], (int, float)) or not 0.0 <= verdict["score"] <= 1.0:
                    verdict["score"] = 0.0
                return verdict
            last_error = verdict["reason"]
            if attempt < _JUDGE_MAX_ATTEMPTS - 1:
                time.sleep(0.5 * (attempt + 1))
        return {"score": 0.0, "reason": last_error}


def build_llm_judge(settings: Settings) -> Optional[JudgeFn]:
    """Build the P6 judge hook from settings, or ``None`` when not enabled.

    Requires ``judge_enabled`` plus a base URL, model, and API key. When any of
    those is missing the judge is disabled and the framework keeps its
    deterministic fail-below-threshold behavior.
    """
    if not settings.judge_enabled:
        return None
    if not settings.judge_base_url or not settings.judge_model or not settings.judge_api_key:
        return None
    return _OpenAIJudge(
        base_url=settings.judge_base_url,
        model=settings.judge_model,
        api_key=settings.judge_api_key,
        timeout=settings.judge_timeout,
        rubric=settings.judge_rubric,
    )
