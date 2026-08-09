"""Trace instrumentation for the "Ask about a location" assistant.

Implements the generic ``TraceCollector`` contract from the AQuA framework
(``aqua.tracing``) plus GeoNames-specific events (``on_plan``,
``on_location_resolved``). The collected logs expose the execution path (which
GeoNames services were called, and in which order) plus the exact parameters
each call received.

Contract (Phase 2): ``AskLocationAssistant.answer()`` returns just the summary
string; tests and the golden-anchor evaluator read telemetry from
``assistant.trace_collector.get_trace_logs()``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aqua.tracing import TraceCollector


class TestTraceCollector(TraceCollector):
    """In-memory collector used by tests and the golden-anchor evaluator."""

    __test__ = False  # not a pytest test class, despite the name

    def __init__(self) -> None:
        self.executed_tools: List[str] = []
        self.executed_tool_calls: List[Dict[str, Any]] = []
        self.tool_outputs: Dict[str, Any] = {}
        self.plan: Dict[str, Any] = {}
        self.location: Dict[str, Any] = {}

    def on_tool_called(
        self, tool_name: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record the call (name and params) and the path position."""
        self.executed_tools.append(tool_name)
        self.executed_tool_calls.append({"name": tool_name, "parameters": params})

    def on_tool_output(self, tool_name: str, data: Dict[str, Any]) -> None:
        """Record the raw JSON returned by the endpoint call."""
        self.tool_outputs[tool_name] = data

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
            "tool_outputs": self.tool_outputs,
            "plan": self.plan,
            "location": self.location,
        }

    def reset(self) -> None:
        """Clear all collected events."""
        self.executed_tools.clear()
        self.executed_tool_calls.clear()
        self.tool_outputs.clear()
        self.plan.clear()
        self.location.clear()
