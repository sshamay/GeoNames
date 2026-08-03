# GeoNames — Python + pytest automation scaffold

A minimal, standards-following pytest project scaffold for the GeoNames workspace.

## Layout

```
GeoNames/
├── config/
│   ├── config.yaml            # single source of truth (local, gitignored)
│   └── config.example.yaml    # committed template
├── src/geonames/              # the package under test / automation library
│   ├── config_loader.py       # load + validate YAML into typed Settings
│   ├── clients/               # thin adapters for external systems (HTTP here)
│   ├── services/              # workflows that orchestrate clients
│   └── models/                # dataclasses for request/response data
├── tests/
│   ├── conftest.py            # shared fixtures (loads config with env="test")
│   ├── data/                  # test-owned sample payloads / expected JSON
│   ├── unit/                  # fast, isolated tests with mocks
│   └── integration/           # tests hitting real/stubbed dependencies
├── pyproject.toml             # packaging (src layout) + pytest config
└── requirements.txt           # dev install entry point (-e .[dev])
```

## Setup (5 minutes)

```bash
# From the repo root - uses the existing Python 3.9 .venv
.venv/bin/pip install -r requirements.txt
```

This installs the `geonames` package in editable mode plus the dev extras
(`pytest`, `pytest-mock`, and runtime deps declared in `pyproject.toml`).

## Running tests

```bash
.venv/bin/python -m pytest                 # full suite
.venv/bin/python -m pytest -m unit         # fast, isolated tests only
.venv/bin/python -m pytest tests/unit/test_config_loader.py -k env
```

## Configuration

All runtime settings live in `config/config.yaml` (env profiles `dev` / `test` /
`staging`, merged over `defaults`). `conftest.py` loads it through the same
`geonames.config_loader.load_config(env="test")` path the rest of the code uses,
so tests never re-declare settings. Never hardcode URLs or secrets in code.

### AQuA P6 LLM-as-a-Judge

The AI-assistant golden-anchor suite evaluates every case through the AQuA
pipeline: **P3** deterministic checks first (structure, keywords, tool-call
params, exact/JSON expected outcome) with fail-fast, cheapest layer first;
**P4** a deterministic `expected_outcome` match; **P5** semantic cosine
similarity (`>= 0.6` passes); then — only when similarity is below 0.6 and the
cheaper layers could not decide — **P6** an optional LLM-as-a-Judge that scores
groundedness/completeness against a structured rubric on a 0-1 scale. The check
id `llm_judge` passes at `score >= 0.3`; below 0.3 the case escalates to
human-in-the-loop.

The judge is **off by default** so the suite stays fully offline and
deterministic. Enable it in `config/config.yaml` (local, gitignored). The
keyless default uses **AI Horde's anonymous tier** — free, no signup, no API
key (the literal key `0000000000`; lowest queue priority):

```yaml
judge_enabled: true
judge_base_url: https://oai.aihorde.net/v1
judge_model: google/gemma-4-31b     # whatever volunteers host on the horde
judge_api_key: "0000000000"         # anonymous access - no registration
judge_timeout: 120
judge_rubric: groundedness_and_completeness
```

Any OpenAI-compatible endpoint works, e.g. Groq (`https://api.groq.com/openai/v1`
+ a free key) or OpenAI (`https://api.openai.com/v1`). The hook is pluggable
(`tests/test_utils/aqua_evaluation.py` → `AQuAEvaluators.llm_judge`): any
callable with signature
`(ai_output, expected_outcome, retrieved_context) -> {"score": 0..1, "reason": str}`
works. When `judge_enabled` is false — or URL/model/key are missing —
`build_llm_judge` returns `None` and below-threshold answers fail exactly as
before. Judge replies are parsed leniently (markdown fences and prose-wrapped
JSON tolerated) and **fail closed** at score 0.0 on transport errors or
unparseable output. `config/config.example.yaml` documents the keys; `config.yaml`
holds the real (local) values.

## Design decisions

- **One config file, not `tests/config/`** — env profiles live in the root
  `config.yaml`; test-owned data (payloads, expected JSON) lives in `tests/data/`.
- **Proper packaging instead of `sys.path` hacks** — `pyproject.toml` uses the
  src layout with `setuptools`; an editable install (`-e .[dev]`) makes
  `import geonames` work everywhere, including PyCharm's test runner.
- **pytest config lives in `pyproject.toml`** (`[tool.pytest.ini_options]`)
  instead of a separate `pytest.ini` — one file for both packaging and tooling.
- **`--strict-markers`** — the `unit` / `integration` markers are registered;
  an unregistered marker fails the run instead of silently passing.
- **Mocking at the boundary** — all mocks use `pytest-mock`'s `mocker` fixture.
  Never third-party mock libraries.
- **Python 3.9 compatible** — `from __future__ import annotations`, typing
  imports, dataclasses; no 3.10+ syntax.

### What I'd add in production

- The actual system-under-test client in `clients/` with per-endpoint methods.
- `pytest -m integration` tests against a stub server, plus `responses`-free
  HTTP mocking via a `_resp()` helper (skill convention: patch at the client
  boundary with `mocker`).
- CI: `pip install -e .[dev] && pytest -m unit` on every push.
- `pytest-cov` + a coverage gate once the real suites exist.
