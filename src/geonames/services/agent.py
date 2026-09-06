"""GeoNames agent: a real LLM tool-calling agent built on LangChain.

Replaces the deterministic ``AskLocationAssistant``. Given a free-text question,
the agent uses an LLM (function calling) to decide which GeoNames tools to call
(geocode, earthquakes, weather), executes them via a tool-calling loop, and
produces a natural-language answer grounded in the fetched data.

It exposes the AQuA golden-anchor SUT contract::

    agent.process_user_query(input) -> str
    agent.trace_collector.get_trace_logs() -> dict

The tools record every invocation into ``GeoNamesTools.ctx``; the agent flushes
those onto its ``trace_collector`` after each query, so the AQuA evaluators see
real execution telemetry from the agent loop.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Optional

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from geonames.services.tools import GeoNamesTools, build_tools
from geonames.tracing import TestTraceCollector, TraceCollector

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are GeoNamesAgent, an assistant that answers questions about a "
    "location using live GeoNames data (recent earthquakes and current "
    "weather observations). "
    "You have exactly these tools: geocode_location(place), "
    "earthquakes(place, date=None, max_rows=None, max_distance_km=None), and "
    "weather(place, max_rows=None). "
    "You MUST follow this tool sequence: first call geocode_location(place) to "
    "resolve the place name, then call the data tool(s) the question actually "
    "asks about: earthquakes(...) for anything about earthquakes, and "
    "weather(...) for anything about weather/rain/temperature/conditions. "
    "If the question asks about BOTH earthquakes and weather, call BOTH data "
    "tools - do not skip either one. "
    "Never stop after geocoding: until you have called the data tool(s) the "
    "question requires, keep calling tools. " 
    "When the user names a distance radius (e.g. 'within 50 miles' or 'within "
    "20 km'), pass max_distance_km to the earthquakes tool (miles * 1.609 = "
    "km), so the quakes are limited to that radius. "
    "If the user refers to their own location ('near me', 'my location', "
    "'my local weather', 'here'), treat it as the default location and pass "
    "that place to the tool (geocode_location/weather/earthquakes resolve "
    "'near me' to it automatically) - do not call the tool with the literal "
    "string 'near me'. "
    "When a radius is used, the tool's summary already reflects only the "
    "quakes inside that radius. If the summary says 'There are N earthquakes', "
    "report that N and never turn it into 'no earthquakes'. The phrase 'within "
    "50 miles' in the user question does NOT mean there are no quakes - the "
    "tool output is the source of truth, not the distance wording. "
    "Never answer from memory or guess: base your answer ONLY on what the "
    "tools returned. If you call a data tool and it returns zero results, say "
    "so explicitly ('no recent earthquakes', 'no weather observations'). "
    "The weather tool returns a 'count' field: when count >= 1, weather "
    "observations EXIST and you MUST report them (e.g. 'N weather "
    "observations near <place>'), even if individual fields look like 'n/a'. "
    "Never say 'no weather observations' when the tool's count is greater "
    "than zero - that contradicts the fetched data. "
    "The earthquakes tool returns a summary that states how many recent "
    "earthquakes were found (e.g. 'There are 5 recent earthquakes near "
    "<place>'). Trust that number verbatim. Do NOT judge recency yourself "
    "from the quake datetimes: even if the returned dates look old to you, "
    "if the tool says there are N recent earthquakes, report N. Only say "
    "'no recent earthquakes' when the tool output literally says there are "
    "none. Never replace a positive tool count with 'no recent earthquakes' "
    "just because the datetimes are not recent in your opinion. "
    "Do not fabricate counts, magnitudes, or observations that no tool "
    "reported. "
    "Today's real current date is {today}. When a query asks for 'today' or "
    "'yesterday' (or similar relative dates), use the real current date "
    "(YYYY-MM-DD) for the earthquakes date parameter - never invent a date. "
    "IMPORTANT: only pass the date parameter to the earthquakes tool when the "
    "user explicitly names a calendar date or a relative day ('today', "
    "'yesterday', 'on 2026-06-15'). For a plain 'recent' or 'near <place>' "
    "earthquake query, do NOT pass a date - leave it out so the most recent "
    "quakes are returned. "
    "Answer concisely in plain English based ONLY on the data the tools "
    "returned."
)


class GeoNamesAgent:
    """LLM tool-calling agent over the GeoNames services."""

    def __init__(
        self,
        tools: GeoNamesTools,
        llm: Any,
        trace_collector: Optional[TraceCollector] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
    ) -> None:
        self._tools = tools
        self._llm = llm
        self.trace_collector = trace_collector or TestTraceCollector()
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._max_iterations = max_iterations
        self._executor = self._build_executor()
        self._usage = {"prompts_processed": 0, "tool_calls": 0, "failed_queries": 0}

    def _build_executor(self) -> AgentExecutor:
        """Assemble the LangChain tool-calling agent + executor."""
        system = (
            self._system_prompt.format(today=date.today().isoformat())
            if "{today}" in self._system_prompt
            else self._system_prompt
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        lc_tools = build_tools(self._tools)
        agent = create_tool_calling_agent(self._llm, lc_tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=lc_tools,
            max_iterations=self._max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )
        return executor

    # -- AQuA SUT contract ------------------------------------------------------

    def process_user_query(self, user_input: str) -> str:
        """Answer a free-text geography question using the agent loop."""
        self.trace_collector.reset()
        self._tools.ctx.reset()
        self._usage["prompts_processed"] += 1
        try:
            result = self._executor.invoke({"input": user_input})
            output = result.get("output", "")
        except Exception as exc:  # agent loop errors fail safe, not crashy
            logger.warning("Agent invocation failed: %s", exc)
            self._usage["failed_queries"] += 1
            output = (
                "Sorry, I could not answer that right now - the GeoNames "
                "request failed."
            )
        self._usage["tool_calls"] += len(self._tools.ctx.calls)
        self._flush_trace(user_input)
        output = f"{output}{self._format_usage()}"
        return output

    def _format_usage(self) -> str:
        """Compact, deterministic usage-statistics block appended to the reply.

        Rendered as a stable ``[usage]`` footer so it is both human-readable and
        machine-assertable by the golden-anchor harness (e.g. via required
        keywords like ``prompts_processed``).
        """
        return (
            "\n[usage] "
            f"prompts_processed={self._usage['prompts_processed']} "
            f"tool_calls={self._usage['tool_calls']} "
            f"failed_queries={self._usage['failed_queries']}"
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return a copy of the agent's cumulative usage counters.

        Counters are session-scoped to this agent instance and accumulate across
        every ``process_user_query`` call:

        - ``prompts_processed``: total queries answered so far.
        - ``tool_calls``: total GeoNames tool invocations across all queries.
        - ``failed_queries``: queries that ended in an agent-loop error.
        """
        return dict(self._usage)

    def _flush_trace(self, user_input: str) -> None:
        """Push the recorded tool calls + outputs onto the trace collector."""
        collector = self.trace_collector
        for call in self._tools.ctx.calls:
            name = call.get("name")
            collector.on_tool_called(name, call.get("parameters"))
        for name, data in self._tools.ctx.outputs.items():
            collector.on_tool_output(name, data)
        location = None
        if "geocode_location" in self._tools.ctx.outputs:
            loc = self._tools.ctx.outputs["geocode_location"]
            location = {"name": loc.get("name"), "lat": loc.get("lat"), "lng": loc.get("lng")}
        collector.on_plan({"question": user_input, "agent": "geonames"})
        collector.on_usage(self.get_usage_stats())
        if location:
            collector.on_location_resolved(location)

    def answer(self, user_input: str) -> str:
        """Alias for :meth:`process_user_query`."""
        return self.process_user_query(user_input)
