# AQuA — Golden-Anchor Evaluation Framework

A portable, project-agnostic AI quality-engineering toolkit for golden-anchor
testing. It evaluates an AI assistant (the *system under test*) against a set
of JSON "golden anchor" cases, using deterministic quality gates plus an
optional LLM-as-a-judge, and produces KPI reports and an HTML dashboard.

It ships as a pytest plugin, so once installed it auto-registers the harness in
any project. No host-project code or config leaks into the framework.

## Installation

```bash
# From this repository (editable, for development)
pip install -e ./aqua

# From a built wheel / PyPI
pip install aqua

# Dev dependencies (pytest, pytest-mock) for running the framework's own tests
pip install -e "./aqua[dev]"
```

Requires Python >= 3.9. Pytest is a peer dependency of the consuming project.

## Quick start

### 1. Create a golden-anchor cases file

Copy `sample_golden_anchor.json` from this package to your project (the pytest
plugin looks for `tests/data/golden_anchor.json` by default) and adapt it. The
sample ships inside the installed package:

```bash
python -c "import aqua, pathlib; print(pathlib.Path(aqua.__file__).parent / 'sample_golden_anchor.json')"
```

Every entry has the same schema:

| Field | Type | Meaning |
|-------|------|---------|
| `case_id` | `str` | Unique id used in pytest node ids, reports and the ledger |
| `scenario` | `str` | Human-readable description |
| `user_input` | `str` | Prompt sent to the assistant |
| `required_keywords` | `list[str]` | Substrings that must appear in the reply (`content_rules`) |
| `forbidden_keywords` | `list[str]` | Substrings that must NOT appear in the reply |
| `required_tools` | `list[{name, parameters}]` | Tools that must be called with these parameters (`agent_logic`) |
| `forbidden_tools` | `list[str]` | Tool names that must NOT be called |
| `expected_outcome` | `str` | Golden reference answer used by the semantic / judge layers |
| `threshold` | `float` | Semantic-similarity threshold (0..1) for `expected_outcome` |
| `simulated_output` | `str` (optional) | Overrides the assistant reply (used for hallucination-demo cases) |

### 2. Provide the SUT and adapter fixtures

In your project's `tests/conftest.py`, implement the host-contract fixtures.
Only `ai_assistant` is mandatory:

```python
import pytest

@pytest.fixture
def ai_assistant():
    """The system under test."""
    return MyAssistant()  # must expose process_user_query() and trace_collector
```

The SUT contract mirrors the framework's `TraceCollector`:
- `ai_assistant.process_user_query(user_input) -> result`
- `ai_assistant.trace_collector.get_trace_logs()` returns a dict with
  `executed_tools`, `executed_tool_calls` (`[{name, parameters}]`), and
  `tool_outputs` (raw fetched data, used by the hallucination gate).

### 2b. Optional: one-place host config (`aqua_config.py`)

Copy the annotated template to your project root — it's THE one place to see
and set everything the framework uses:

```bash
python -c "import aqua, pathlib; print(pathlib.Path(aqua.__file__).parent / 'aqua_config.example.py')"
cp <that path> aqua_config.py
```

The plugin loads it automatically. Everything is optional; anything you set here
takes precedence over the `AQUA_*` env vars (which remain only as fallbacks):

| Knob | Meaning | Fallback |
|------|---------|----------|
| `CASES_PATH` | Golden-anchor cases JSON | `AQUA_CASES` env / `tests/data/golden_anchor.json` |
| `DATA_DIR` | Raw data dir for `load_cases()` | `AQUA_DATA_DIR` env / `tests/data` |
| `REPORT_DIR` | Run reports + dashboard dir | `AQUA_REPORT_DIR` env / `reports` |
| `THRESHOLDS` | `aqua.config.Thresholds` (semantic/judge/case gates) | framework defaults (see below) |
| `JUDGE` | `aqua.config.JudgeConfig` (see below) | `AQUA_JUDGE_*` env vars |
| `hallucination_extractor` | `callable(ai_output, tool_outputs) -> list[str]` | gate disabled |

That table is the **complete** configuration surface — every tunable knob. The
env vars in the tables below are only fallbacks for the same knobs; you do not
need to set any of them when the file is present. Two things deliberately stay
**outside** the file because they are code/data, not configuration: the SUT
itself (`ai_assistant` fixture in `tests/conftest.py`) and the golden-anchor
cases JSON (its per-case `threshold` field overrides `THRESHOLDS.default_case`
for that case). The conftest fixtures below can still override any of this
per-project.

### 3. Write the golden-anchor test file

A single thin test that the plugin parametrizes over your cases file:

```python
import pytest
from aqua.golden_anchor import assert_golden_anchor, run_golden_anchor_case

pytestmark = pytest.mark.ai_assistant


def test_golden_anchor_case(ai_assistant, aqua_evaluators_class, run_ledger, case_and_id):
    eval_result, ai_output = run_golden_anchor_case(
        ai_assistant, aqua_evaluators_class, run_ledger, case_and_id)
    assert_golden_anchor(eval_result, ai_output)
```

`run_golden_anchor_case` runs the assistant, evaluates it, and records the
outcome in the session ledger; `assert_golden_anchor` fails the test on
evaluation failure (and skips when the case has no coverage).

### 4. Run

```bash
# Default cases file: $AQUA_CASES or tests/data/golden_anchor.json
pytest tests/ai_assistant/test_golden_anchor_eval.py

# Point at a different cases file
pytest --aqua-cases path/to/cases.json

# Run a single case
pytest -k "GA-001"
```

The plugin generates one pytest case per JSON entry (`case_and_id`
parametrization), and at the end of the session writes the KPI run report
(`reports/aqua_run_<timestamp>.json`, `reports/latest.json`,
`reports/history.jsonl`) and renders `reports/dashboard.html`.

## Judge configuration

The judge is disabled by default, keeping the suite offline and deterministic.
Configure it in `aqua_config.py` as `JUDGE = JudgeConfig(...)` (see the
template), or via `AQUA_JUDGE_*` env vars, or by overriding the
`aqua_judge_config` fixture. The precedence is: host `aqua_config.JUDGE` >
env vars > framework defaults.

| Env var | Default | Meaning |
|---------|---------|---------|
| `AQUA_JUDGE_ENABLED` | `false` | Master switch |
| `AQUA_JUDGE_PROVIDER` | `openai_compatible` | `openai_compatible`, `openai`, `aihorde`, or `ollama` |
| `AQUA_JUDGE_MODEL` | *(none)* | Model name (required) |
| `AQUA_JUDGE_BASE_URL` | *(provider default)* | OpenAI-compatible base URL |
| `AQUA_JUDGE_API_KEY` | *(provider default)* | Bearer API key |
| `AQUA_JUDGE_TIMEOUT` | `60` | HTTP timeout seconds |
| `AQUA_JUDGE_RUBRIC` | `groundedness_and_completeness` | Rubric injected into the judge prompt |
| `AQUA_JUDGE_DEBUG` | `false` | Attach raw judge output (`raw`) to verdicts |
| `AQUA_JUDGE_SYSTEM_PROMPT` | *(framework default)* | Override the judge's system prompt (project-specific scoring instructions) |
| `AQUA_JUDGE_MAX_TOKENS` | `256` | Cap on judge model output tokens |
| `AQUA_JUDGE_MAX_ATTEMPTS` | `3` | Retries on transport errors / unparseable verdicts |

`JudgeConfig` fields map 1:1 to these env vars (`enabled`, `provider`, `model`,
`base_url`, `api_key`, `timeout`, `rubric`, `debug`, `system_prompt`,
`max_tokens`, `max_attempts`).

## Thresholds

Evaluation gates are tuned with `aqua.config.Thresholds` (set `THRESHOLDS` in
your `aqua_config.py`; the `aqua_thresholds` fixture applies them). Defaults:

| Threshold | Default | Meaning |
|-----------|---------|---------|
| `expected_outcome_semantic` | `0.6` | Cosine similarity needed for the cheap semantic layer (P5) to pass without escalation |
| `llm_judge_pass` | `0.3` | Minimum P6 judge score for the `llm_judge` check to pass |
| `default_case` | `0.9` | Aggregate confidence below which a case fails / escalates to HITL (used when the case JSON has no `threshold`) |

A case's own `threshold` field overrides `default_case` for that case.

## Hallucination gate

WHAT COUNTS AS A HALLUCINATION is defined by **you**, in the same one-place
`aqua_config.py`. The framework only provides the mechanism: a case FAILS the
`hallucination_check` gate when the extractor returns a non-empty list of
contradictions between the assistant's reply and the data it actually fetched.

Set `hallucination_extractor` to a function:

```python
def hallucination_extractor(ai_output, tool_outputs) -> list[str]:
    # ai_output     = the assistant's reply text (string)
    # tool_outputs  = raw data the trace recorded from the fetchers, keyed by endpoint
    # return [] when the reply is consistent with the data, else a list of
    # human-readable mismatch descriptions, e.g. ["count 50 != 3"]
```

- Empty list → the gate PASSES (no contradictions).
- Non-empty list → the gate FAILS with the exact mismatches in the report.
- Omit the function (or return `None`) → the gate is disabled.

The template ships a worked number-claim checker to adapt. Example: a reply
"Found 50 earthquakes" when the fetched data has 3 → extractor returns
`["50 earthquakes != 3"]` → `hallucination_check FAILED`. The gate only runs
for cases that actually fetched data (`tool_outputs` non-empty).

Provider defaults for `base_url` / `api_key` (only the model is always
required):

| Provider | Base URL | API key |
|----------|----------|---------|
| `openai_compatible` | *(required)* | *(required)* |
| `openai` | `https://api.openai.com/v1` | *(required)* |
| `aihorde` | `https://oai.aihorde.net/v1` | `0000000000` (anonymous) |
| `ollama` | `http://localhost:11434/v1` | `ollama` |

The judge talks to any OpenAI-compatible `/chat/completions` endpoint. When it
is disabled or underconfigured, the evaluators fail closed (semantic similarity
fallback only). Unknown provider names raise `ValueError` instead of silently
disabling the judge.

## Reports & CLI

Reports land in `$AQUA_REPORT_DIR` (default `reports/`). The `aqua` CLI
regenerates them from an existing run without re-running tests:

```bash
aqua report               # print KPIs from reports/latest.json
aqua report --run <path>  # print KPIs from a specific run JSON
aqua dashboard            # render reports/dashboard.html
```

## Framework layout

```
aqua/
├── pyproject.toml            # package config (pytest11 entry point, aqua CLI)
├── README.md
├── src/aqua/
│   ├── __init__.py
│   ├── config.py             # JudgeConfig + AQUA_JUDGE_* env reading
│   ├── evaluation.py         # AQuAEvaluators (quality gates + run_case)
│   ├── golden_anchor.py      # run_golden_anchor_case / assert_golden_anchor
│   ├── judge.py              # LLM-as-a-judge client + build_llm_judge
│   ├── loaders.py            # load_cases (AQUA_DATA_DIR)
│   ├── plugin.py             # pytest plugin (host config, parametrization, ledger, reports)
│   ├── reporting.py          # AQuARunLedger (KPIs, run reports, history)
│   ├── tracing.py            # TraceCollector contract
│   ├── dashboard.py          # HTML dashboard renderer
│   ├── cli.py                # `aqua report` / `aqua dashboard`
│   ├── aqua_config.example.py    # annotated host-config template -> copy to project root
│   └── sample_golden_anchor.json # copy into your project
└── tests/                    # framework's own unit tests
```

## Running the framework's own tests

```bash
pip install -e "./aqua[dev]"
pytest aqua/tests -m unit
```

