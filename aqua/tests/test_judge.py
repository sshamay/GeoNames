"""Unit tests for the AQuA P6 LLM-as-a-Judge client (aqua.judge)."""

import pytest

from aqua.config import JudgeConfig
from aqua.judge import _extract_verdict, build_llm_judge


def _judge_config(**overrides):
    base = dict(
        enabled=True,
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="judge-model",
        api_key="secret",
        timeout=30.0,
        rubric="groundedness_and_completeness",
    )
    base.update(overrides)
    return JudgeConfig(**base)


@pytest.mark.unit
def test_build_llm_judge_returns_none_when_disabled():
    assert build_llm_judge(_judge_config(enabled=False)) is None


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["base_url", "model", "api_key"])
def test_build_llm_judge_returns_none_when_underconfigured(missing):
    assert build_llm_judge(_judge_config(**{missing: None})) is None


@pytest.mark.unit
def test_build_llm_judge_returns_callable_when_configured():
    judge = build_llm_judge(_judge_config())
    assert callable(judge)


@pytest.mark.unit
def test_build_llm_judge_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown judge_provider 'nonexistent'"):
        build_llm_judge(_judge_config(provider="nonexistent"))


@pytest.mark.unit
def test_ollama_provider_applies_defaults(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"score": 0.5, "reason": "ok"}'}}]
    }
    post = mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_judge_config(
        provider="ollama", base_url=None, api_key=None))
    assert callable(judge)
    judge("reply", "golden", [])

    call = post.call_args
    assert call.args[0] == "http://localhost:11434/v1/chat/completions"
    assert call.kwargs["headers"]["Authorization"] == "Bearer ollama"


@pytest.mark.unit
def test_ollama_provider_requires_model():
    assert build_llm_judge(_judge_config(
        provider="ollama", base_url=None, api_key=None, model=None)) is None


@pytest.mark.unit
def test_aihorde_provider_applies_defaults(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"score": 0.5, "reason": "ok"}'}}]
    }
    post = mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_judge_config(
        provider="aihorde", base_url=None, api_key=None))
    assert callable(judge)
    judge("reply", "golden", [])

    call = post.call_args
    assert call.args[0] == "https://oai.aihorde.net/v1/chat/completions"
    assert call.kwargs["headers"]["Authorization"] == "Bearer 0000000000"


@pytest.mark.unit
def test_openai_provider_requires_api_key():
    assert build_llm_judge(_judge_config(
        provider="openai", base_url=None, api_key=None)) is None


@pytest.mark.unit
def test_openai_provider_applies_default_base_url(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"score": 0.5, "reason": "ok"}'}}]
    }
    post = mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_judge_config(
        provider="openai", base_url=None, api_key="sk-real"))
    assert callable(judge)
    judge("reply", "golden", [])

    call = post.call_args
    assert call.args[0] == "https://api.openai.com/v1/chat/completions"
    assert call.kwargs["headers"]["Authorization"] == "Bearer sk-real"


@pytest.mark.unit
def test_judge_posts_to_chat_completions_and_parses_verdict(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"score": 0.9, "reason": "grounded"}'}}]
    }
    post = mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_judge_config())
    verdict = judge("There were 3 earthquakes.", "There were 3 earthquakes.", [{"n": 3}])

    assert verdict == {"score": 0.9, "reason": "grounded"}
    call = post.call_args
    assert call.kwargs["json"]["model"] == "judge-model"
    assert call.kwargs["json"]["temperature"] == 0
    assert call.kwargs["json"]["max_tokens"] == 256
    assert call.kwargs["headers"]["Authorization"] == "Bearer secret"
    assert call.args[0] == "https://api.example.com/v1/chat/completions"


@pytest.mark.unit
def test_judge_uses_custom_system_prompt_when_configured(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"score": 0.9, "reason": "grounded"}'}}]
    }
    post = mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_judge_config(system_prompt="You are the ACME judge."))
    judge("reply", "golden", [])

    messages = post.call_args.kwargs["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "You are the ACME judge."}


@pytest.mark.unit
def test_judge_uses_default_system_prompt_when_unset(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"score": 0.9, "reason": "grounded"}'}}]
    }
    post = mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_judge_config())
    judge("reply", "golden", [])

    messages = post.call_args.kwargs["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert "strict quality judge" in messages[0]["content"]


@pytest.mark.unit
def test_judge_debug_mode_attaches_raw_output(mocker):
    content = '{"score": 0.9, "reason": "grounded"}'
    response = mocker.Mock()
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_judge_config(debug=True))
    verdict = judge("reply", "golden", [])

    assert verdict["score"] == 0.9
    assert verdict["reason"] == "grounded"
    assert verdict["raw"] == content


@pytest.mark.unit
def test_judge_debug_mode_attaches_raw_on_unparseable(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "this is not json"}}]
    }
    mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_judge_config(debug=True))
    verdict = judge("reply", "golden", [])

    assert verdict["score"] == 0.0
    assert "unparseable" in verdict["reason"]
    assert verdict["raw"] == "this is not json"


@pytest.mark.unit
def test_judge_accepts_fenced_json_output(mocker):
    response = mocker.Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '```json\n{"score": 0.7, "reason": "ok"}\n```'}}]
    }
    mocker.patch("requests.post", return_value=response)

    judge = build_llm_judge(_judge_config())
    verdict = judge("reply", "golden", [])
    assert verdict["score"] == 0.7


@pytest.mark.unit
def test_judge_fails_closed_on_transport_error(mocker):
    mocker.patch("requests.post", side_effect=RuntimeError("connection refused"))

    judge = build_llm_judge(_judge_config())
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

    judge = build_llm_judge(_judge_config())
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

    judge = build_llm_judge(_judge_config())
    verdict = judge("reply", "golden", [])
    assert verdict["score"] == 0.75
