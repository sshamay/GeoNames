"""LLM client abstraction for the "Ask about a location" assistant.

The assistant is model-agnostic: it talks to an :class:`LlmClient`, selected
per environment. ``create_llm_client`` chooses the implementation from an
explicit provider name or the ``GEONAMES_LLM_PROVIDER`` environment variable,
so the stub can be swapped for a real model later without touching the
assistant or its tests.

The stub is a deterministic stand-in: it reads the GeoNames JSON and renders a
natural-language summary without any network call, so the full pipeline
(parse -> fetch -> summarize) is testable offline.
"""

from __future__ import annotations

import os
from typing import List, Optional, Protocol, Sequence

from geonames.models.assistant import EndpointResult


class LlmClient(Protocol):
    """The seam a real model provider will implement."""

    def summarize(
        self,
        question: str,
        results: Sequence[EndpointResult],
        location: Optional[str] = None,
    ) -> str:
        """Render a natural-language summary of the GeoNames results."""
        ...


class StubLlmClient:
    """Deterministic, offline LLM stand-in used in tests and dev."""

    def summarize(
        self,
        question: str,
        results: Sequence[EndpointResult],
        location: Optional[str] = None,
    ) -> str:
        summaries: List[str] = []
        for result in results:
            if result.endpoint == "earthquakes":
                summaries.append(_summarize_earthquakes(result.data))
            elif result.endpoint == "weather":
                summaries.append(_summarize_weather(result.data))
        if not summaries:
            return "No GeoNames data was relevant to the question."
        where = f" near {location}" if location else ""
        return f"Found {_join(summaries)}{where}."


class RealLlmClient:
    """Placeholder for the production LLM call (not implemented yet).

    Keeps the ``real`` provider ready in the factory so wiring it up later is
    a drop-in change - implement ``summarize`` against the model of choice.
    """

    def summarize(
        self,
        question: str,
        results: Sequence[EndpointResult],
        location: Optional[str] = None,
    ) -> str:
        raise NotImplementedError(
            "RealLlmClient is a placeholder; switch GEONAMES_LLM_PROVIDER "
            "back to 'stub' or implement summarize() to use a real model."
        )


def create_llm_client(provider: Optional[str] = None) -> LlmClient:
    """Build an LLM client, resolved from ``provider``, then the environment.

    Args:
        provider: Explicit provider name (``stub`` | ``real``). When omitted,
            the ``GEONAMES_LLM_PROVIDER`` env var is used, defaulting to
            ``stub``.

    Returns:
        The selected LlmClient implementation.

    Raises:
        ValueError: For an unknown provider name.
    """
    selected = provider or os.environ.get("GEONAMES_LLM_PROVIDER") or "stub"
    if selected == "stub":
        return StubLlmClient()
    if selected == "real":
        return RealLlmClient()
    raise ValueError(f"Unknown LLM provider '{selected}'. Known: stub, real")


def _summarize_earthquakes(data: dict) -> str:
    quakes = data.get("earthquakes", []) or []
    if not quakes:
        return "no recent earthquakes"
    strongest = max(quake["magnitude"] for quake in quakes)
    return f"{len(quakes)} recent earthquakes (strongest magnitude {strongest:.1f})"


def _summarize_weather(data: dict) -> str:
    observations = data.get("weatherObservations", []) or []
    if not observations:
        return "no weather observations"
    return f"weather observations from {len(observations)} stations"


def _join(parts: List[str]) -> str:
    """Join summary fragments grammatically (empty-safe, no extra 'and' noise)."""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]
