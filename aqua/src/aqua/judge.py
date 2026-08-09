"""LLM-as-a-Judge client for the golden-anchor evaluation.

Implements the ``AQuAEvaluators.llm_judge`` hook signature::

    callable(ai_output, expected_outcome, retrieved_context)
        -> {"score": float (0..1), "reason": str}

Talks to any OpenAI-compatible ``/chat/completions`` endpoint. Build the hook
from a :class:`aqua.config.JudgeConfig` with :func:`build_llm_judge`, which
returns ``None`` when the feature is disabled or underconfigured, so the suite
stays fully offline and deterministic by default.

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

from aqua.config import JudgeConfig

JudgeFn = Callable[[str, str, List[Any]], Dict[str, Any]]

_UNPARSEABLE_PREFIX = "Judge returned unparseable output:"

# The default scoring instructions. A project may override this via
# JudgeConfig.system_prompt / AQUA_JUDGE_SYSTEM_PROMPT; the override replaces
# the whole prompt. Kept public so a host config can show or reuse it.
DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are a strict quality judge for an AI assistant. Score the assistant's "
    "reply for groundedness in the retrieved data and completeness against the "
    "golden reference answer. Output exactly one JSON object and nothing else "
    "- no prose, no markdown fences: {\"score\": 0.0-1.0, \"reason\": "
    "\"short rationale\"}."
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

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float, rubric: str, debug: bool = False, system_prompt: Optional[str] = None, max_tokens: int = 256, max_attempts: int = 3):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._rubric = rubric
        self._debug = debug
        self._system_prompt = system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts

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
        last_content = None
        for attempt in range(self._max_attempts):
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
                        "max_tokens": self._max_tokens,
                        "messages": [
                            {"role": "system", "content": self._system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=self._timeout,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                last_content = content
            except Exception as exc:
                last_error = f"LLM judge call failed: {exc}"
                if attempt < self._max_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return self._with_debug({"score": 0.0, "reason": last_error}, None)

            verdict = _extract_verdict(content)
            if not verdict["reason"].startswith(_UNPARSEABLE_PREFIX):
                if not isinstance(verdict["score"], (int, float)) or not 0.0 <= verdict["score"] <= 1.0:
                    verdict["score"] = 0.0
                return self._with_debug(verdict, content)
            last_error = verdict["reason"]
            if attempt < self._max_attempts - 1:
                time.sleep(0.5 * (attempt + 1))
        return self._with_debug({"score": 0.0, "reason": last_error}, last_content)

    def _with_debug(self, verdict: Dict[str, Any], raw: Optional[str]) -> Dict[str, Any]:
        """Attach the raw model output when judge_debug is enabled."""
        if self._debug:
            verdict["raw"] = raw
        return verdict


# Provider names each judge factory registers under. The default
# "openai_compatible" covers any Bearer-auth /chat/completions endpoint
# (AI Horde, Ollama, local proxies) but requires explicit base_url/model/key.
_JUDGE_PROVIDER_DEFAULTS = {
    "openai_compatible": {"base_url": None, "api_key": None},
    "openai": {"base_url": "https://api.openai.com/v1", "api_key": None},
    "aihorde": {"base_url": "https://oai.aihorde.net/v1", "api_key": "0000000000"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
}


def _build_judge(config: JudgeConfig, provider: str) -> Optional[JudgeFn]:
    """Build an OpenAI-compatible judge for ``provider`` using its defaults.

    Provider-specific base_url/api_key defaults apply when the config fields
    are left empty; the model is always required (there is no sane default).
    Returns None (fail closed) when the judge is underconfigured.
    """
    defaults = _JUDGE_PROVIDER_DEFAULTS[provider]
    base_url = config.base_url or defaults["base_url"]
    api_key = config.api_key or defaults["api_key"]
    if not base_url or not config.model or not api_key:
        return None
    return _OpenAIJudge(
        base_url=base_url,
        model=config.model,
        api_key=api_key,
        timeout=config.timeout,
        rubric=config.rubric,
        debug=config.debug,
        system_prompt=config.system_prompt,
        max_tokens=config.max_tokens,
        max_attempts=config.max_attempts,
    )


def build_llm_judge(config: JudgeConfig) -> Optional[JudgeFn]:
    """Build the P6 judge hook from a JudgeConfig, or ``None`` when not enabled.

    ``config.provider`` selects which provider factory builds the client; each
    provider applies its own base_url/api_key defaults, so only the model (and
    any non-default values) need to be set. An unknown provider name is a
    config error and raises rather than silently disabling the judge.

    Requires ``config.enabled``. When the selected provider is underconfigured
    the judge is disabled and the framework keeps its deterministic
    fail-below-threshold behavior.
    """
    if not config.enabled:
        return None
    if config.provider not in _JUDGE_PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown judge_provider '{config.provider}'. "
            f"Known providers: {sorted(_JUDGE_PROVIDER_DEFAULTS)}"
        )
    return _build_judge(config, config.provider)
