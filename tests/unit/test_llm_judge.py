"""Unit tests for the LLM-as-a-Judge client (tests/test_utils/llm_judge.py)."""

import pytest

from geonames.config_loader import Settings
from test_utils.llm_judge import _extract_verdict, build_llm_judge


def _settings(**overrides):
    base = dict(
        app_name="geonames",
        env="test",
        judge_enabled=True,
        judge_base_url="https://api.example.com/v1",
        judge_model="judge-model",
        judge_api_key="secret",
        judge_timeout=30.0,
        judge_rubric="groundedness_and_completeness",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.mark.unit
def test_build_llm_judge_returns_none_when_disabled():
    assert build_llm_judge(_settings(judge_enabled=False)) is None


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["judge_base_url", "judge_model", "judge_api_key"])
def test_build_llm_judge_returns_none_when_underconfigured(missing):
    assert build_llm_judge(_settings(**{missing: None})) is None


@pytest.mark.unit
def test_build_llm_judge_returns_callable_when_configured():
    judge = build_llm_judge(_settings())
    assert callable(judge)


@pytest.mark.unit
def test_judge_posts_to_chat_completions_and_parses_verdict(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"score": 0.9, "reason": "grounded"}'}}]
    }
    post = mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_settings())
    verdict = judge("There were 3 earthquakes.", "There were 3 earthquakes.", [{"n": 3}])

    assert verdict == {"score": 0.9, "reason": "grounded"}
    call = post.call_args
    assert call.kwargs["json"]["model"] == "judge-model"
    assert call.kwargs["json"]["temperature"] == 0
    assert call.kwargs["json"]["max_tokens"] == 256
    assert call.kwargs["headers"]["Authorization"] == "Bearer secret"
    assert call.args[0] == "https://api.example.com/v1/chat/completions"


@pytest.mark.unit
def test_judge_accepts_fenced_json_output(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '```json\n{"score": 0.7, "reason": "ok"}\n```'}}]
    }
    mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_settings())
    verdict = judge("reply", "golden", [])
    assert verdict["score"] == 0.7


@pytest.mark.unit
def test_judge_fails_closed_on_transport_error(mocker):
    mocker.patch("requests.post", side_effect=RuntimeError("connection refused"))

    judge = build_llm_judge(_settings())
    verdict = judge("reply", "golden", [])
    assert verdict["score"] == 0.0
    assert "LLM judge call failed" in verdict["reason"]


@pytest.mark.unit
def test_judge_clamps_out_of_range_score(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"score": 1.5, "reason": "great"}'}}]
    }
    mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_settings())
    verdict = judge("reply", "golden", [])
    assert verdict["score"] == 0.0


@pytest.mark.unit
def test_extract_verdict_fails_closed_on_garbage():
    verdict = _extract_verdict("no json here")
    assert verdict["score"] == 0.0
    assert "unparseable" in verdict["reason"]


@pytest.mark.unit
def test_extract_verdict_tolerates_prose_wrapped_json():
    content = (
        "Based on the analysis, the answer is correct.\n"
        '{"score": 0.8, "reason": "matches golden"}\nHope this helps.'
    )
    verdict = _extract_verdict(content)
    assert verdict["score"] == 0.8
    assert verdict["reason"] == "matches golden"


@pytest.mark.unit
def test_extract_verdict_tolerates_missing_reason():
    verdict = _extract_verdict('{"score": 1}')
    assert verdict["score"] == 1.0
    assert verdict["reason"] == ""


@pytest.mark.unit
def test_extract_verdict_tolerates_fenced_and_prose_json(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "```json\n{\"score\": 0.75}\n```"}}]
    }
    mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_settings())
    verdict = judge("reply", "golden", [])
    assert verdict["score"] == 0.75
