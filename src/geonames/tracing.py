"""Trace instrumentation for the "Ask about a location" assistant.

Implements the generic ``TraceCollector`` contract from the AQuA framework
(``aqua.tracing``) plus GeoNames-specific events (``on_plan``,
``on_location_resolved``). The collected logs expose the execution path (which
GeoNames services were called, and in which order) plus the exact parameters
each call received.

A ``tracer`` (an OpenTelemetry ``Tracer``) may be injected: when present every
emitted event also opens a span, giving deep OTel telemetry (AQuA observe pillar)
in addition to the in-memory trace logs. When no tracer is given (the default)
the collector is purely in-memory and offline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aqua.tracing import TraceCollector


class TestTraceCollector(TraceCollector):
    """In-memory collector used by tests and the golden-anchor evaluator."""

    __test__ = False  # not a pytest test class, despite the name

    def __init__(self, tracer=None) -> None:
        self.executed_tools: List[str] = []
        self.executed_tool_calls: List[Dict[str, Any]] = []
        self.tool_outputs: Dict[str, Any] = {}
        self.plan: Dict[str, Any] = {}
        self.location: Dict[str, Any] = {}
        self.usage: Dict[str, Any] = {}
        self._tracer = tracer

    def on_tool_called(
        self, tool_name: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record the call (name and params) and the path position."""
        self.executed_tools.append(tool_name)
        self.executed_tool_calls.append({"name": tool_name, "parameters": params})
        if self._tracer is not None:
            with self._tracer.start_as_current_span(
                f"tool_call:{tool_name}"
            ) as span:
                span.set_attribute("aqua.tool.name", tool_name)
                if params:
                    span.set_attribute(
                        "aqua.tool.params", _stringify(params)
                    )

    def on_tool_output(self, tool_name: str, data: Dict[str, Any]) -> None:
        """Record the raw JSON returned by the endpoint call."""
        self.tool_outputs[tool_name] = data
        if self._tracer is not None:
            span = self._tracer.start_span(f"tool_output:{tool_name}")
            span.set_attribute("aqua.tool.name", tool_name)
            span.end()

    def on_plan(self, plan: Dict[str, Any]) -> None:
        """Record the parsed query plan."""
        self.plan = dict(plan)

    def on_location_resolved(self, location: Dict[str, Any]) -> None:
        """Record the resolved location."""
        self.location = dict(location)

    def on_usage(self, usage: Dict[str, Any]) -> None:
        """Record a snapshot of the assistant's cumulative usage counters."""
        self.usage = dict(usage)

    def get_trace_logs(self) -> Dict[str, Any]:
        """Return the accumulated trace logs (golden-anchor compatible).

        ``retrieved_context`` is derived from the raw tool outputs so the P6
        LLM-as-a-judge has the actual fetched data to grade the reply against.
        """
        return {
            "retrieved_context": list(self.tool_outputs.values()),
            "executed_tools": self.executed_tools,
            "executed_tool_calls": self.executed_tool_calls,
            "tool_outputs": self.tool_outputs,
            "plan": self.plan,
            "location": self.location,
            "usage": self.usage,
        }

    def reset(self) -> None:
        """Clear all collected events (usage counters are NOT cumulative-reset)."""
        self.executed_tools.clear()
        self.executed_tool_calls.clear()
        self.tool_outputs.clear()
        self.plan.clear()
        self.location.clear()
        self.usage.clear()


def _stringify(value: Any) -> str:
    """Compact string form of a params dict for OTel attributes."""
    if isinstance(value, dict):
        return ",".join(f"{k}={_stringify(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ";".join(_stringify(v) for v in value)
    return str(value)
