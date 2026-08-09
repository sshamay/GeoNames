"""AQuA host configuration - THE one place to configure the framework.

Copy this file to the root of your project as ``aqua_config.py`` and adapt it.
The AQuA pytest plugin loads it automatically; every knob is optional. Anything
you leave out falls back to an ``AQUA_*`` environment variable, then to the
framework default (see ``aqua/README.md`` for the full env-var reference).
This file takes precedence over env vars, so configure here and nowhere else.

There are four kinds of things you can set here:

  1. Paths (CASES_PATH / DATA_DIR / REPORT_DIR)
  2. Evaluation thresholds (THRESHOLDS)
  3. The LLM-as-a-judge settings (JUDGE - every AQUA_JUDGE_* knob)
  4. The hallucination-gate definition (hallucination_extractor) - here you
     decide what counts as a hallucination for your reply format.

The SUT itself (``ai_assistant``) cannot be generic - provide it as a fixture
in your project's ``tests/conftest.py``.
"""

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
# Golden-anchor cases JSON: one pytest case is generated per entry.
CASES_PATH = "tests/data/golden_anchor.json"

# Where load_cases() looks for raw data, if your cases reference any.
DATA_DIR = "tests/data"

# Where run reports (aqua_run_*.json, latest.json, history.jsonl) and the
# HTML dashboard are written.
REPORT_DIR = "reports"

# ---------------------------------------------------------------------------
# 2. Evaluation thresholds
# ---------------------------------------------------------------------------
from aqua.config import Thresholds

THRESHOLDS = Thresholds(
    expected_outcome_semantic=0.6,  # P5 semantic similarity gate (0..1)
    llm_judge_pass=0.3,             # minimum P6 judge score to pass (0..1)
    default_case=0.9,               # aggregate confidence that triggers HITL/failure
)


# ---------------------------------------------------------------------------
# 3. LLM-as-a-judge
# ---------------------------------------------------------------------------
# The judge is DISABLED by default so the suite stays offline and
# deterministic. Enable it only for cases the deterministic and semantic
# layers cannot decide.
#
# Every field below is the config-file equivalent of one AQUA_JUDGE_* env var
# (AQUA_JUDGE_ENABLED, AQUA_JUDGE_PROVIDER, ...). The file wins when set;
# the env var is only a fallback. Configure here and nowhere else.
from aqua.config import JudgeConfig

JUDGE = JudgeConfig(
    enabled=False,          # ↔ AQUA_JUDGE_ENABLED   set True to enable the judge
    provider="ollama",      # ↔ AQUA_JUDGE_PROVIDER  openai_compatible|openai|aihorde|ollama
    model="llama3.2:3b",    # ↔ AQUA_JUDGE_MODEL     required; no sane default
    base_url=None,          # ↔ AQUA_JUDGE_BASE_URL  None = provider default (see README)
    api_key=None,           # ↔ AQUA_JUDGE_API_KEY   None = provider default
    timeout=60.0,           # ↔ AQUA_JUDGE_TIMEOUT
    rubric="groundedness_and_completeness",  # ↔ AQUA_JUDGE_RUBRIC
    debug=False,            # ↔ AQUA_JUDGE_DEBUG    attach raw judge output
    system_prompt=None,     # ↔ AQUA_JUDGE_SYSTEM_PROMPT (override judge instructions)
    max_tokens=256,         # ↔ AQUA_JUDGE_MAX_TOKENS   cap on judge output
    max_attempts=3,         # ↔ AQUA_JUDGE_MAX_ATTEMPTS retries on bad verdicts
)


# ---------------------------------------------------------------------------
# 4. Hallucination gate (optional, off by default)
# ---------------------------------------------------------------------------
# WHAT COUNTS AS A HALLUCINATION is defined HERE, by you. A case is flagged as
# hallucinated when the extractor returns a NON-EMPTY list of contradictions
# between the assistant's reply and the data it actually fetched.
#
# The extractor is called as:  hallucination_extractor(ai_output, tool_outputs)
#   - ai_output    the assistant's reply text (the string the SUT returned)
#   - tool_outputs the raw data the trace recorded from the fetchers, keyed by
#                  endpoint (whatever your trace_collector stores)
# and must return a list[str] of human-readable mismatches (empty = consistent).
# Return None / omit the function to disable the gate.
#
# Minimal worked example: flag every number claim in the reply that the fetched
# data does not support. Adapt the regexes + data keys to YOUR reply format.
import re

_NUMBER_CLAIM = re.compile(r"(\d+)\s+(\w+)")  # e.g. "5 earthquakes", "3 results"


def hallucination_extractor(ai_output, tool_outputs):
    """List number claims in the reply that the fetched data contradicts.

    This demo compares each "N word" claim against the matching key in
    tool_outputs (e.g. tool_outputs['earthquakes'] == ['…', '…']). A real
    extractor encodes the exact claims your summarize() step can make.
    """
    issues = []
    text = (ai_output or "").lower()
    for endpoint, data in tool_outputs.items():
        if not isinstance(data, (list, dict)):
            continue
        expected = len(data)
        for count, noun in _NUMBER_CLAIM.findall(text):
            if noun.rstrip("s") != endpoint.rstrip("s"):
                continue
            if int(count) != expected:
                issues.append(f"{count} {noun} != fetched {expected}")
    return issues
