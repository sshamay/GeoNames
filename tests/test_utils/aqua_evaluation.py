import json
import math
from collections import Counter

try:
    from pydantic import BaseModel, ValidationError as _PydanticValidationError
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False

# AQuA detect.md P5: compare the generated answer against the golden
# reference answer using semantic (cosine) similarity. Threshold follows the
# skill's example (0.85). Below it, a richer layer (e.g. an embedding model)
# should decide instead of the cheap matcher.
EXPECTED_OUTCOME_SEMANTIC_THRESHOLD = 0.85


def _normalize(text):
    return " ".join(text.lower().split())


def _extract_content(ai_output):
    """
    Extract the user-facing text from a structured assistant payload.

    Many assistants serialize their reply as JSON with a "content" field
    (SmartSpend returns SmartSpendResponse). Text-based checks (content_rules,
    expected_outcome) run against that content. Non-JSON or schemaless output
    passes through unchanged.
    """
    if not isinstance(ai_output, str):
        return ai_output
    try:
        data = json.loads(ai_output)
    except Exception:
        return ai_output
    if isinstance(data, dict) and isinstance(data.get("content"), str):
        return data["content"]
    return ai_output


def _char_ngrams(text, n=3):
    """Dependency-free embedding fallback: character n-gram term counts."""
    normalized = " ".join(text.lower().split())
    if len(normalized) < n:
        return Counter({normalized: 1}) if normalized else Counter()
    return Counter(normalized[i:i + n] for i in range(len(normalized) - n + 1))


def _embed(text):
    """
    Pluggable embedder. Swap in a real embedding provider here when available
    (e.g. sentence-transformers, fastembed, or an embeddings API). Falls back
    to character n-gram vectors so the framework stays portable and offline.
    """
    return _char_ngrams(text)


def _cosine(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    norm_a = math.sqrt(sum(c * c for c in vec_a.values()))
    norm_b = math.sqrt(sum(c * c for c in vec_b.values()))
    if not norm_a or not norm_b:
        return 0.0
    dot = sum(vec_a[gram] * vec_b[gram] for gram in vec_a if gram in vec_b)
    return dot / (norm_a * norm_b)


class AQuAEvaluators:
    """
    A modular evaluation engine based on the AQuA framework.
    It generically maps Golden Anchor requirements to specific quality gates [7].
    """

    @staticmethod
    def evaluate_execution_path(trace_logs, required_documents):
        """
        Pillar 5: Deep Pipeline Tracing.
        Catches the 'Perfect Outcome, Broken Process' trap by verifying context usage [2, 8].

        Behavior:
        - If trace_logs contains retrieved_context with data, validate presence of required_documents and fail if missing.
        - If required_documents are declared but retrieved_context is empty/missing, return FAILED (should have retrieved docs).
        """
        trace_logs = trace_logs or {}
        # If required_documents are declared but retrieved_context is empty/missing, fail
        # (should have retrieved required docs, but didn't)
        if "retrieved_context" not in trace_logs or not trace_logs.get("retrieved_context"):
            return {
                "check_name": "execution_path",
                "status": "FAILED",
                "score": 0.0,
                "reason": f"Required documents {required_documents} were not retrieved [2]."
            }

        retrieved = trace_logs.get("retrieved_context", [])
        # Ensure all required documents were actually used in the trace
        missing = [doc for doc in required_documents if doc not in retrieved]

        if missing:
            return {
                "check_name": "execution_path",
                "status": "FAILED",
                "score": 0.0,
                "reason": f"Pipeline is Blind: Missing documents {missing} [2]."
            }
        return {"check_name": "execution_path", "status": "PASSED", "score": 1.0}

    @staticmethod
    def evaluate_structural_compliance(ai_output, model=None, schema_keys=None):
        """
        Layer 1: Rule-Based Validation (AQuA detect.md P4).
        Verifies the AI reliably outputs strict formats like JSON [9, 10].

        Behavior:
        - Non-JSON output -> FAILED.
        - If a Pydantic BaseModel is provided, validate the parsed payload
          against it (model_validate); a ValidationError -> FAILED with
          per-field error details. Otherwise -> PASSED.
        - Without a model, optionally require a set of top-level keys
          (legacy schema_keys behavior).
        """
        try:
            data = json.loads(ai_output)
        except (json.JSONDecodeError, TypeError):
            return {"check_name": "structural_compliance", "status": "FAILED", "score": 0.0,
                    "reason": "Output is not valid JSON."}

        if model is not None:
            if not _PYDANTIC_AVAILABLE:
                return {"check_name": "structural_compliance", "status": "SKIPPED", "score": None,
                        "reason": "pydantic not installed; structural contract not validated."}
            try:
                model.model_validate(data)
            except _PydanticValidationError as exc:
                errors = "; ".join(f"{'.'.join(str(part) for part in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors())
                return {"check_name": "structural_compliance", "status": "FAILED", "score": 0.0,
                        "reason": f"Output violates contract ({type(model).__name__}): {errors}"}
            return {"check_name": "structural_compliance", "status": "PASSED", "score": 1.0}

        if schema_keys and not all(key in data for key in schema_keys):
            return {"check_name": "structural_compliance", "status": "FAILED", "score": 0.0,
                    "reason": "Incomplete JSON structure [9]."}
        return {"check_name": "structural_compliance", "status": "PASSED", "score": 1.0}

    @staticmethod
    def evaluate_content_rules(ai_output, required_keywords=None, forbidden_keywords=None):
        """
        Layer 1: Deterministic Content Checks.
        Validates keyword presence and prevents 'vibe coding' hallucinations [9, 11].
        """
        ai_output = ai_output or ""
        reasons = []
        score = 1.0

        if required_keywords:
            missing = [word for word in required_keywords if word.lower() not in ai_output.lower()]
            if missing:
                score = 0.0
                reasons.append(f"Missing required keywords: {missing}")

        if forbidden_keywords:
            found = [word for word in forbidden_keywords if word.lower() in ai_output.lower()]
            if found:
                score = 0.0
                reasons.append(f"Contains forbidden keywords: {found}")

        return {"check_name": "content_rules", "status": "PASSED" if score == 1.0 else "FAILED", "score": score, "reason": "; ".join(reasons)}

    @staticmethod
    def evaluate_agent_logic(trace_logs, required_tools=None, forbidden_tools=None):
        """
        Pillar 2: Agentic Execution Logic.
        Validates that tools are invoked in the correct order and parameters [7, 12, 13].

        Behavior:
        - If trace_logs contains executed_tools, perform checks.
        - If trace_logs lacks executed_tools (no instrumentation), return SKIPPED when required_tools/forbidden_tools are present.
        """
        # Only SKIP when instrumentation is genuinely absent (no trace keys).
        # If a collector is present (even with an empty executed_tools), the
        # assistant really ran but failed to call required tools -> FAILED.
        trace_logs = trace_logs or {}
        instrumented = "executed_tools" in trace_logs or bool(trace_logs.get("executed_tool_calls"))
        if not instrumented and (required_tools or forbidden_tools):
            return {"check_name": "agent_logic", "status": "SKIPPED", "score": None, "reason": "No executed_tools trace available; agent_logic not run."}

        executed_tools = trace_logs.get("executed_tools", [])
        tool_calls = trace_logs.get("executed_tool_calls") or []
        reasons = []
        score = 1.0

        def _names(tools):
            return [t.get("name", t) if isinstance(t, dict) else t for t in tools]

        def _call_matches(call, required):
            if call["name"] != required["name"]:
                return False
            required_params = required.get("parameters")
            if not required_params:
                return True
            actual_params = call.get("parameters") or {}
            if not actual_params:
                return False
            return all(actual_params.get(k) == v for k, v in required_params.items())

        def _executed(required):
            name = required["name"] if isinstance(required, dict) else required
            if isinstance(required, dict) and required.get("parameters") and tool_calls:
                return any(_call_matches(c, required) for c in tool_calls)
            return name in executed_tools

        required_names = _names(required_tools or [])
        forbidden_names = _names(forbidden_tools or [])

        if required_names:
            missing = [t for t in required_tools if not _executed(t)]
            if missing:
                score = 0.0
                reasons.append(f"Required tools not called with expected parameters: {missing}")

        if forbidden_names:
            found = [t for t, name in zip(forbidden_tools, forbidden_names) if name in executed_tools]
            if found:
                score = 0.0
                reasons.append(f"Forbidden tools were called: {found}")

        return {"check_name": "agent_logic", "status": "PASSED" if score == 1.0 else "FAILED", "score": score, "reason": "; ".join(reasons)}

    @staticmethod
    def evaluate_expected_outcome(ai_output, expected_outcome):
        """
        Layer X: Expected outcome checking.
        Deterministic exact match first (P4); falls back to JSON structural
        match, then semantic (cosine) similarity (P5).
        """
        if expected_outcome is None or expected_outcome == "":
            # No expectation declared: flag it so empty expectations are not
            # silently treated as passing (counts toward MISSING_COVERAGE).
            return {"check_name": "expected_outcome", "status": "SKIPPED", "score": None, "reason": "No expected outcome declared."}

        # Deterministic fast path (P4): exact normalized equality needs no
        # embedding/cosine machinery.
        if _normalize(ai_output) == _normalize(expected_outcome):
            return {"check_name": "expected_outcome", "status": "PASSED", "score": 1.0}

        # Attempt structured JSON comparison if possible
        try:
            expected_json = json.loads(expected_outcome)
            ai_json = json.loads(ai_output)

            def match(expected, actual):
                if isinstance(expected, dict) and isinstance(actual, dict):
                    for k, v in expected.items():
                        if k not in actual or not match(v, actual[k]):
                            return False
                    return True
                if isinstance(expected, list) and isinstance(actual, list):
                    return all(any(match(e, a) for a in actual) for e in expected)
                return expected == actual

            ok = match(expected_json, ai_json)
            if ok:
                return {"check_name": "expected_outcome", "status": "PASSED", "score": 1.0}
            return {"check_name": "expected_outcome", "status": "FAILED", "score": 0.0, "reason": "Structured JSON expected outcome not satisfied."}
        except Exception:
            # Semantic layer (AQuA detect.md P5): cosine similarity between the
            # AI output and the golden reference answer. Rejects degenerate
            # fragments ("n", "I found ") without a length heuristic.
            similarity = _cosine(_embed(ai_output), _embed(expected_outcome))
            if similarity >= EXPECTED_OUTCOME_SEMANTIC_THRESHOLD:
                return {"check_name": "expected_outcome", "status": "PASSED", "score": 1.0}
            return {"check_name": "expected_outcome", "status": "FAILED", "score": 0.0,
                    "reason": f"Semantic similarity {similarity:.3f} below threshold {EXPECTED_OUTCOME_SEMANTIC_THRESHOLD}."}

    @classmethod
    def run_case(cls, case, ai_output, trace_logs):
        """
        Generic Runner: Dynamically checks what's relevant based on the Golden Anchor [4, 14].

        Only runs evaluators for checks explicitly requested in the case. If a case
        does not declare a field (e.g., no required_tools), that check is treated
        as not-applicable and is not added to results.

        If the evaluator class defines a response_model (a Pydantic BaseModel),
        the structured output is additionally validated against it. Projects
        without a structured contract can omit response_model and this check
        is skipped entirely.
        """
        results = []

        # 0. Check Structural Compliance (structured JSON output contract).
        # Only runs when the project configures a response_model on the class.
        response_model = getattr(cls, "response_model", None)
        if response_model is not None:
            results.append(cls.evaluate_structural_compliance(ai_output, model=response_model))

        # Text-based checks run against the extracted user-facing content
        # (unwraps SmartSpendResponse-style JSON payloads when present).
        content = _extract_content(ai_output)

        # 1. Check Execution Path (Required Documents)
        if case.get("required_documents"):
            results.append(cls.evaluate_execution_path(trace_logs, case["required_documents"]))

        # 2. Check Content (Keywords)
        if case.get("required_keywords") or case.get("forbidden_keywords"):
            results.append(
                cls.evaluate_content_rules(content, case.get("required_keywords"), case.get("forbidden_keywords")))

        # 3. Check Agentic Logic (Tools)
        if case.get("required_tools") or case.get("forbidden_tools"):
            results.append(
                cls.evaluate_agent_logic(trace_logs, case.get("required_tools"), case.get("forbidden_tools")))

        # 4. Check Expected Outcome
        if "expected_outcome" in case:
            results.append(cls.evaluate_expected_outcome(content, case.get("expected_outcome")))

        # 5. Final Risk-Based Confidence Gate
        return cls.calculate_confidence(results, case.get("threshold", 0.90))

    @staticmethod
    def calculate_confidence(results, threshold):
        """
        Pillar 3: Risk-Based Confidence Scoring.
        Replaces binary pass/fail with structured confidence signals [15, 16].
        Now ignores SKIPPED checks when computing averages and returns NO_TESTS_RUN
        when there are no numeric results.
        """
        if not results:
            return {"is_safe": False, "aggregate_score": 0.0, "action": "NO_TESTS_RUN", "details": results}

        numeric_results = [r for r in results if isinstance(r.get('score'), (int, float))]
        skipped_count = sum(1 for r in results if r.get('score') is None)

        if not numeric_results:
            return {"is_safe": False, "aggregate_score": 0.0, "action": "NO_TESTS_RUN", "details": results, "skipped": skipped_count}

        avg_score = sum(r['score'] for r in numeric_results) / len(numeric_results)

        # If any checks were skipped, consider this a coverage failure even if numeric
        # checks pass. This forces test owners to address coverage gaps.
        if skipped_count > 0:
            return {
                "is_safe": False,
                "aggregate_score": avg_score,
                "action": "MISSING_COVERAGE",
                "details": results,
                "skipped": skipped_count
            }

        return {
            "is_safe": avg_score >= threshold,
            "aggregate_score": avg_score,
            "action": "RELEASE" if avg_score >= threshold else "ESCALATE_TO_HITL",
            "details": results,
            "skipped": skipped_count
        }
