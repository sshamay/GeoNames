# GeoNames AI Assistant

A pytest automation scaffold for testing a GeoNames-based AI assistant that answers location questions about earthquakes and weather.

## Quick Start

```bash
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
│       └── golden_anchor.json  # AQuA test cases (GN-001 to GN-020)
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

The project uses AQuA (Architecting Quality for AI Applications) methodology with golden anchor test cases:

- **GN-001 to GN-020**: Test cases covering earthquakes, weather queries, location resolution, and AI quality gates
- Each case verifies: tool calls, parameters, output keywords, and semantic correctness
- Run with: `pytest tests/ai_assistant/test_golden_anchor_eval.py -v`

## Configuration

1. Copy `config/config.example.yaml` to `config/config.yaml`
2. Add your GeoNames username and API settings
3. Optionally configure LLM judge endpoint (AI Horde default provided)

## Dashboard

After running golden anchor tests, generate the dashboard:

```bash
python scripts/render_dashboard.py
# Open reports/dashboard.html
```

The dashboard shows:
- KPI cards (pass rate, HITL count, judge escalations)
- Trend charts across all test runs
- Per-check pass rates
- Failed case details

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Fast, isolated tests with mocks (no network) |
| `integration` | Tests with real or stubbed dependencies |
| `services` | Tests exercising GeoNames service endpoints |
| `user_flow` | End-to-end multi-service tests |
| `ai_assistant` | Golden anchor evaluation tests |
