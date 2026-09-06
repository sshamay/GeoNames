"""Unit tests for the GeoNames agent's cumulative usage statistics.

The agent exposes session-scoped counters (prompts_processed, tool_calls,
failed_queries) via ``get_usage_stats()`` and appends a stable ``[usage]``
footer to every reply so the golden-anchor harness can assert them.

These tests run offline: the LangChain executor is stubbed so no LLM or
network call is made.
"""

import pytest

from geonames.config_loader import Settings
from geonames.factory import build_assistant


class _FakeClient:
    """Duck-typed GeoNamesClient that must never be reached at runtime."""

    def get(self, endpoint, params):
        raise AssertionError(f"unexpected network call to {endpoint}")


def _make_assistant(monkeypatch):
    settings = Settings(
        app_name="geonames",
        env="test",
        geonames_username="testuser",
        geonames_base_url="https://secure.geonames.org",
        geonames_timeout=10.0,
        assistant_max_rows=3,
        default_location="Sacramento",
    )
    assistant = build_assistant(settings, client=_FakeClient())

    class _StubExecutor:
        def __init__(self):
            self.calls = {"n": 0}

        def invoke(self, payload):
            self.calls["n"] += 1
            return {"output": f"fixture answer {self.calls['n']}"}

    stub = _StubExecutor()
    monkeypatch.setattr(assistant, "_executor", stub)
    return assistant, stub


@pytest.mark.unit
def test_prompts_processed_accumulates_across_queries(monkeypatch):
    assistant, _ = _make_assistant(monkeypatch)

    assert assistant.get_usage_stats()["prompts_processed"] == 0
    assistant.process_user_query("first")
    assistant.process_user_query("second")
    assistant.process_user_query("third")

    stats = assistant.get_usage_stats()
    assert stats["prompts_processed"] == 3
    assert stats["failed_queries"] == 0


@pytest.mark.unit
def test_reply_embeds_usage_footer(monkeypatch):
    assistant, _ = _make_assistant(monkeypatch)

    out = assistant.process_user_query("what is the weather?")
    assert "[usage]" in out
    assert "prompts_processed=1" in out
    assert "tool_calls=" in out
    assert "failed_queries=0" in out
    # The footer is appended to the answer, not replacing it.
    assert out.startswith("fixture answer 1")


@pytest.mark.unit
def test_get_usage_stats_returns_copy(monkeypatch):
    assistant, _ = _make_assistant(monkeypatch)
    assistant.process_user_query("q")
    stats = assistant.get_usage_stats()
    stats["prompts_processed"] = 999  # mutating the returned dict must not leak
    assert assistant.get_usage_stats()["prompts_processed"] == 1


@pytest.mark.unit
def test_failed_query_is_counted(monkeypatch):
    assistant, _ = _make_assistant(monkeypatch)

    def boom(payload):
        raise RuntimeError("agent loop failed")

    monkeypatch.setattr(assistant._executor, "invoke", boom)
    out = assistant.process_user_query("boom")

    assert out.startswith("Sorry, I could not answer")
    stats = assistant.get_usage_stats()
    assert stats["failed_queries"] == 1
    assert stats["prompts_processed"] == 1
