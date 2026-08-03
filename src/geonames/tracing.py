"""Trace instrumentation for the "Ask about a location" assistant.

Mirrors the TraceCollector pattern used by the SmartSpend golden-anchor
framework so the same generic evaluator can verify this assistant: the
collected logs expose the execution path (which GeoNames services were called,
and in which order) plus the exact parameters each call received.

Contract (Phase 2): ``AskLocationAssistant.answer()`` returns just the summary
string; tests and the golden-anchor evaluator read telemetry from
``assistant.trace_collector.get_trace_logs()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TraceCollector(ABC):
    """Collects events emitted by the assistant during execution.

    Implementations are decoupled from assistant logic so multiple collectors
    can coexist (test, monitoring, file logging).
    """

    @abstractmethod
    def on_tool_called(
        self, tool_name: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a GeoNames service endpoint call.

        Args:
            tool_name: Endpoint name, e.g. ``earthquakes`` or ``weather``.
            params: The parameters the service was called with.
        """
        ...

    @abstractmethod
    def get_trace_logs(self) -> Dict[str, Any]:
        """Return the accumulated trace logs.

        Keys mirror the golden-anchor evaluator contract:
        - ``executed_tools``: endpoint names in call order (the execution path)
        - ``executed_tool_calls``: one ``{"name": ..., "parameters": ...}`` per call
        - ``retrieved_context``: always empty (GeoNames fetches, not documents)
        """
        ...

    def reset(self) -> None:
        """Clear trace logs between queries."""


class TestTraceCollector(TraceCollector):
    """In-memory collector used by tests and the golden-anchor evaluator."""

    __test__ = False  # not a pytest test class, despite the name

    def __init__(self) -> None:
        self.executed_tools: List[str] = []
        self.executed_tool_calls: List[Dict[str, Any]] = []
        self.plan: Dict[str, Any] = {}
        self.location: Dict[str, Any] = {}

    def on_tool_called(
        self, tool_name: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record the call (name and params) and the path position."""
        self.executed_tools.append(tool_name)
        self.executed_tool_calls.append({"name": tool_name, "parameters": params})

    def on_plan(self, plan: Dict[str, Any]) -> None:
        """Record the parsed query plan."""
        self.plan = dict(plan)

    def on_location_resolved(self, location: Dict[str, Any]) -> None:
        """Record the resolved location."""
        self.location = dict(location)

    def get_trace_logs(self) -> Dict[str, Any]:
        """Return the accumulated trace logs (golden-anchor compatible)."""
        return {
            "retrieved_context": [],
            "executed_tools": self.executed_tools,
            "executed_tool_calls": self.executed_tool_calls,
            "plan": self.plan,
            "location": self.location,
        }

    def reset(self) -> None:
        """Clear all collected events."""
        self.executed_tools.clear()
        self.executed_tool_calls.clear()
        self.plan.clear()
        self.location.clear()


class NoOpTraceCollector(TraceCollector):
    """Zero-overhead collector for production (does nothing)."""

    def on_tool_called(
        self, tool_name: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    def get_trace_logs(self) -> Dict[str, Any]:
        return {
            "retrieved_context": [],
            "executed_tools": [],
            "executed_tool_calls": [],
            "plan": {},
            "location": {},
        }

    def reset(self) -> None:
        pass
