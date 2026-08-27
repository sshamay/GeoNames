"""Generic trace collector contract for the AQuA framework.

The evaluators read the execution path from ``get_trace_logs()``. Host
projects implement this ABC in their own telemetry (e.g. a ``TestTraceCollector``
that also records project-specific events).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class TraceCollector(ABC):
    """Collects events emitted by an AI assistant during execution.

    Implementations are decoupled from assistant logic so multiple collectors
    can coexist (test, monitoring, file logging).
    """

    @abstractmethod
    def on_tool_called(
        self, tool_name: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an external service/tool call.

        Args:
            tool_name: The tool/endpoint name, e.g. ``weather``.
            params: The parameters the tool was called with.
        """
        ...

    @abstractmethod
    def on_tool_output(self, tool_name: str, data: Dict[str, Any]) -> None:
        """Record the raw output returned by a tool call.

        Kept separate from :meth:`on_tool_called` so the execution path stays
        readable while raw responses stay available for KPI checks (e.g. the
        hallucination gate compares numbers in the reply against this data).
        """
        ...

    @abstractmethod
    def get_trace_logs(self) -> Dict[str, Any]:
        """Return the accumulated trace logs.

        Keys mirror the golden-anchor evaluator contract:
        - ``executed_tools``: tool names in call order (the execution path)
        - ``executed_tool_calls``: one ``{"name": ..., "parameters": ...}`` per call
        - ``tool_outputs``: raw output JSON per tool (for KPI checks)
        - ``retrieved_context``: retrieved documents/context (may be empty)
        """
        ...

    def reset(self) -> None:
        """Clear trace logs between queries."""
