"""AQuA P6 LLM-as-a-Judge client for the golden-anchor evaluation.

Implements the ``AQuAEvaluators.llm_judge`` hook signature::

    callable(ai_output, expected_outcome, retrieved_context)
        -> {"score": float (0..1), "reason": str}

Talks to any OpenAI-compatible ``/chat/completions`` endpoint. Build the hook
from settings with :func:`build_llm_judge`, which returns ``None`` when the
feature is disabled or underconfigured, so the suite stays fully offline and
deterministic by default.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

import requests

from geonames.config_loader import Settings

JudgeFn = Callable[[str, str, List[Any]], Dict[str, Any]]

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict quality judge for an AI assistant that answers questions "
    "about a location (recent earthquakes, strongest magnitude, weather station "
    "observations). Score the assistant's reply for groundedness in the "
    "retrieved data and completeness against the golden reference answer. "
    "Respond with strict JSON only: "
    "{\"score\": 0.0-1.0, \"reason\": \"short rationale\"}."
)


def _extract_verdict(content: str) -> Dict[str, Any]:
    """Parse ``{"score": ..., "reason": ...}`` from model output.

    Tolerates markdown-fenced JSON. Returns ``{"score": 0.0, ...}`` when the
    output cannot be parsed so an uncooperative judge fails closed.
    """
    text = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        payload = json.loads(text)
        return {"score": float(payload["score"]), "reason": str(payload.get("reason", ""))}
    except Exception:
        return {"score": 0.0, "reason": "Judge returned unparseable output: " + text[:200]}


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
            return {"score": 0.0, "reason": f"LLM judge call failed: {exc}"}

        verdict = _extract_verdict(content)
        if not isinstance(verdict["score"], (int, float)) or not 0.0 <= verdict["score"] <= 1.0:
            verdict["score"] = 0.0
        return verdict


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
