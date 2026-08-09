# GeoNames AI Assistant

A pytest automation scaffold for testing a GeoNames-based AI assistant that answers location questions about earthquakes and weather.

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run unit tests (fast, no network)
pytest tests/unit/ -m unit

# Run golden anchor tests
pytest tests/ai_assistant/test_golden_anchor_eval.py

# Run specific golden anchor cases
pytest tests/ai_assistant/test_golden_anchor_eval.py -k GN-019

# Run all tests
pytest

# Run with verbose output
pytest -v
```

## Configuration

1. Copy `config/config.example.yaml` to `config/config.yaml`
2. Add your GeoNames username and API settings
3. Optionally configure the LLM judge. Pick a provider via `judge_provider`; each has its own defaults for `judge_base_url`/`judge_api_key`, so only `judge_model` (and any non-default values) need to be set:
   - `judge_provider: aihorde` — keyless anonymous access at `https://oai.aihorde.net/v1`, API key `"0000000000"`, e.g. model `google/gemma-4-31b`
   - `judge_provider: ollama` (recommended for offline/fast/privacy-safe runs) — defaults to `http://localhost:11434/v1` / `ollama`; run `ollama pull llama3.2:3b` once
   - `judge_provider: openai` — defaults to `https://api.openai.com/v1`; needs a real API key
   - `judge_provider: openai_compatible` (default) — any other Bearer-auth `/chat/completions` endpoint; `judge_base_url` and `judge_api_key` are required

## Dashboard

The dashboard is generated automatically at the end of every golden anchor test run (`reports/dashboard.html`). Open it with:

```bash
open reports/dashboard.html (using prefered browser)
```

To regenerate the dashboard from an existing run without re-running tests:

```bash
python scripts/render_dashboard.py
```

The dashboard shows:
- KPI cards (pass rate, HITL count, judge escalations)
- Trend charts across all test runs
- Per-check pass rates
- Failed case details

## Project Structure

```
GeoNames/
├── src/geonames/
│   ├── clients/        # GeoNames HTTP client
│   ├── config_loader.py # Configuration loading from YAML
│   ├── factory.py       # Assistant factory function
│   ├── models/          # Data models (Location, AssistantPlan, etc.)
│   └── services/        # Core services (intent parser, assistant, APIs)
├── tests/
│   ├── unit/            # Fast unit tests (mocks, no network)
│   ├── services/        # Service-level integration tests
│   ├── user_flows/      # End-to-end user journey tests
│   ├── ai_assistant/    # Golden anchor evaluation framework
│   ├── test_utils/      # Shared test utilities and AQuA framework
│   ├── conftest.py      # Pytest fixtures
│   └── data/
│       └── golden_anchor.json  # AQuA test cases (GN-001 to GN-025)
├── scripts/
│   └── render_dashboard.py  # Generate KPI HTML dashboard
├── config/
│   ├── config.yaml          # Local config (gitignored)
│   └── config.example.yaml  # Template config
└── reports/                # Generated test reports (gitignored)
```

## Dependencies

| Category | Packages |
|----------|----------|
| Runtime | PyYAML, requests, urllib3<2 |
| Test | pytest, pytest-mock, pydantic |

## AQuA Golden Anchor Testing

This project demonstrates a layered quality engineering approach for evaluating AI-powered applications, following the AQuA (Architecting Quality for AI Applications) methodology. For a detailed explanation, see the [AQuA Framework Medium article](https://medium.com/@shay.shamay/the-aqua-framework-architecting-quality-for-ai-applications-de61b533e406).

### Overview

The framework evaluates the complete AI execution lifecycle:

- **Intent understanding** — does the assistant correctly parse the user's question?
- **Tool selection & execution** — are the right tools called with correct parameters?
- **Business rule compliance** — required/ forbidden keywords, tool gating
- **Response quality & groundedness** — factual accuracy, hallucination detection
- **Risk-based escalation** — deterministic first, then semantic evaluation, then HITL

### Demonstration Approach

Some scenarios are intentionally configured to fail to demonstrate the framework's ability to:

- Detect AI quality issues
- Provide visibility into individual quality dimensions
- Generate actionable failure diagnostics

In production, these signals drive continuous improvement through prompt refinement, orchestration optimization, dataset evolution, and regression prevention.

### Quality Philosophy

Traditional automation asks:

> "Did the system return the expected output?"

AI quality engineering asks:

> "Did the AI system behave correctly, safely, and according to business expectations?"

The framework combines multiple evaluation layers:

| Layer | Purpose |
|---|---|
| **Deterministic Checks** | Validate structure, rules, and contracts |
| **Expected Outcome** | Validate meaning and relevance (cosine similarity >= 0.6) |
| **LLM-as-a-Judge** | Evaluate complex quality dimensions (P6 semantic evaluation) |
| **Hallucination Gate** | Detect fabricated or unsupported data |
| **Human-in-the-Loop** | Handle high-risk uncertainty (Escalate to HITL) |

- **GN-001 to GN-025**: Test cases covering earthquakes, weather queries, location resolution, and AI quality gates
- Each case verifies: tool calls, parameters, output keywords, and semantic correctness
- Run with: `pytest tests/ai_assistant/test_golden_anchor_eval.py -v`

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Fast, isolated tests with mocks (no network) |
| `integration` | Tests with real or stubbed dependencies |
| `services` | Tests exercising GeoNames service endpoints |
| `user_flow` | End-to-end multi-service tests |
| `ai_assistant` | Golden anchor evaluation tests |

---

## Test Case Coverage

The project uses a tiered testing methodology with three layers:

### Unit Tests (97 cases)
Fast, isolated tests with mocked dependencies - no network calls required. Covers:
- Question parsing and location extraction
- Assistant orchestration with fake fetchers
- Configuration loading
- AQuA evaluator logic
- LLM judge client behavior

### Service Tests (data-driven)
Integration tests that exercise real GeoNames API endpoints:

| Test File | Cases | Description |
|-----------|-------|-------------|
| `test_earthquakes.py` | 4 bbox + 4 filter + 1 auth | Bounding box queries, filter validation, auth error handling |
| `test_ground_truth_earthquakes.py` | 2 historic events | Cross-reference against USGS & EMSC catalogs (Tohoku 2011, Sumatra 2012) |
| `test_find_nearby.py` | Live | Toponym proximity lookups |

### User Flow Tests
End-to-end workflows combining multiple services, verifying data flow integrity:

| Case ID | Date | Magnitude | Description |
|---------|------|----------|-------------|
| ridgecrest_2019_07_06 | 2019-07-06 | >=5.5M | M6.4 Ridgecrest earthquake near Los Angeles |
| anchorage_2018_11_30 | 2018-11-30 | >=6.5M | M7.1 Anchorage earthquake near Cook Inlet |
| napa_2014_08_24 | 2014-08-24 | >=5.5M | M6.0 Napa earthquake near San Francisco Bay |

**What user flow tests check (vs golden anchor):**
- Verifies coordinate hand-off: epicenter lat/lng from `earthquakesJSON` are reused verbatim in `findNearbyJSON`
- Validates data integrity: magnitudes within range, coordinates inside bbox, distances within radius
- Asserts auto-report generation matches a deterministic formula (`M{mag:.1f} near {name}`)
- Does NOT use AQuA quality gates or LLM judge; focuses on mechanical correctness of the multi-service pipeline

### AQuA Golden Anchor Cases (GN-001 to GN-025)
AI quality evaluation framework covering real queries, adversarial cases, and escalation:

| Case ID | Name | Category | Location | Status |
|---------|------|----------|----------|--------|
| GN-001 | Earthquakes and weather near Sacramento | Real query | Sacramento, CA | Pass |
| GN-002 | Weather only near San Francisco | Real query | San Francisco, CA | Pass |
| GN-003 | Earthquakes on specific date near Seattle | Real query | Seattle, WA | Pass |
| GN-004 | Earthquakes only near Denver (no weather) | Real query + tool gating | Denver, CO | Pass |
| GN-005 | Earthquakes only near Seattle (today) | Real query + tool gating | Seattle, WA | Pass |
| GN-006 | Weather only near Denver | Real query | Denver, CO | Pass |
| GN-007 | Earthquakes near San Francisco within 50 miles | Real query + radius | San Francisco, CA | Pass |
| GN-008 | Earthquakes on specific date near Los Angeles | Real query + date filter | Los Angeles, CA | Pass |
| GN-009 | Rain near New York must not trigger earthquake tool | Real query + tool gating | New York, NY | Pass |
| GN-011 | Adversarial: missing required keyword | Security/accuracy | New York, NY | Fail (blocked) |
| GN-012 | Adversarial: forbidden keyword present | Security/accuracy | Sacramento, CA | Fail (blocked) |
| GN-013 | Adversarial: required earthquakes tool not called | Security/accuracy | New York, NY | Fail (blocked) |
| GN-014 | Adversarial: forbidden weather tool is called | Security/accuracy | Sacramento, CA | Fail (blocked) |
| GN-015 | Adversarial: semantic mismatch (severe storm vs actual) | Accuracy | Denver, CO | Fail (judge) |
| GN-016 | Greeting only - graceful guidance reply | Graceful fallback | N/A | Pass |
| GN-017 | Greeting then weather query | Combined query | Denver, CO | Pass |
| GN-018 | Adversarial: hallucination demo | Security/accuracy | Seattle, WA | Fail (caught) |
| GN-019 | Complex Data Task: California earthquake + rain yesterday | Integration | California | Pass |
| GN-020 | HITL escalation trigger: semantic mismatch | Escalation | California | Fail (HITL) |
| GN-021 | Ambiguous location: Springfield (default IL) | Location parsing | Springfield, IL | Pass |
| GN-022 | Ambiguous location: Columbus (default OH) | Location parsing | Columbus, OH | Pass |
| GN-023 | State-based location with downtown prefix | Location parsing | Austin, TX | Pass |
| GN-024 | State-based location with yesterday window | Location parsing | Austin, TX | Pass |
| GN-025 | Conversational "near me" defaults to Sacramento | Conversational | Sacramento, CA | Pass |

### AQuA Quality Gates

Each golden anchor case is evaluated against five quality dimensions:

| Gate | Description | How It Works |
|------|-------------|-------------|
| `content_rules` | Required keywords present, forbidden keywords absent | Static string assertion on AI output |
| `agent_logic` | Required tools called with correct params, forbidden tools not called | Compares executed tools vs case spec |
| `expected_outcome` | Semantic similarity between AI output and golden answer | Cosine similarity >=0.6, else LLM judge |
| `llm_judge` | P6 semantic evaluation for borderline cases | AI Horde judge model scores 0-1 |
| `hallucination_check` | Numbers in reply must match fetched data | Claims extractor compares summary vs raw JSON |