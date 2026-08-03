import json
import math
from collections import Counter

try:
    from pydantic import BaseModel, ValidationError as _PydanticValidationError
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False

# AQuA detect.md P5: compare the generated answer against the golden
# reference answer using semantic (cosine) similarity. At or above this the
# cheap matcher is enough. Below it, a richer layer (the P6 LLM judge) should
# decide instead - see EXPECTED_OUTCOME_SEMANTIC_THRESHOLD below.
EXPECTED_OUTCOME_SEMANTIC_THRESHOLD = 0.6

# AQuA detect.md P6: the LLM-as-a-Judge verdict score that counts as a pass
# once the cheap deterministic/semantic layers could not decide. Below this the
# expected-outcome check fails and the run escalates to human-in-the-loop.
LLM_JUDGE_PASS_THRESHOLD = 0.3


def extract_assistant_output(result):
    """
    Extract the reply text from an assistant call result.

    Phase 1: assistant returns (response, trace_logs) tuple
    Phase 2: assistant returns just response string

    This helper makes the harness compatible with both phases.
    """
    return result[0] if isinstance(result, tuple) else result


def failure_summary(eval_result):
    """
    Compact one-line reason for the pytest short summary (the first line of the
    assertion message is the only text pytest shows next to 'FAILED').
    """
    failed = [
        f"{check.get('check_name')} FAILED ({check.get('reason') or 'no reason'})"
        for check in eval_result.get("details") or []
        if check.get("status") == "FAILED"
    ]
    if failed:
        return "; ".join(failed)
    action = eval_result.get("action")
    score = eval_result.get("aggregate_score")
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
    return f"action={action} aggregate_score={score_text}"


def format_eval_result(eval_result):
    """
    Render an AQuA eval result as compact multi-line text focused on the
    failing checks. The first line of the assertion message is the pytest
    short-summary reason; the rest is per-check status + aggregate + AI output.
    """
    if not isinstance(eval_result, dict):
        return str(eval_result)

    score = eval_result.get("aggregate_score")
    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)

    lines = []
    for check in eval_result.get("details") or []:
        status = check.get("status")
        reason = check.get("reason") or ""
        lines.append(f"  {check.get('check_name')}: {status}{' - ' + reason if reason else ''}")

    lines.append(f"  aggregate_score: {score_text} -> {eval_result.get('action')}")
    return "\n".join(lines)


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


def _call_matches_params(call, required):
    """True when a recorded call matches a required tool's name and declared parameters."""
    if call.get("name") != required["name"]:
        return False
    required_params = required.get("parameters")
    if not required_params:
        return True
    actual_params = call.get("parameters") or {}
    if not actual_params:
        return False
    return all(actual_params.get(key) == value for key, value in required_params.items())


def required_tool_executed(required, executed_tools, tool_calls):
    """
    Whether a required tool was actually called, with the right parameters.

    ``required`` is either a tool name or a dict with ``name`` and optional
    ``parameters``. When parameters are declared and a ``tool_calls`` trace is
    present, at least one recorded call must match the name and every declared
    parameter value exactly. Otherwise (no parameters declared, or no call-level
    trace) it degrades to a name-only check against ``executed_tools``.
    """
    name = required["name"] if isinstance(required, dict) else required
    if isinstance(required, dict) and required.get("parameters") and tool_calls:
        return any(_call_matches_params(c, required) for c in tool_calls)
    return name in executed_tools


class AQuAEvaluators:
    """
    A modular evaluation engine based on the AQuA framework.
    It generically maps Golden Anchor requirements to specific quality gates [7].
    """

    # Pluggable project hook: callable(ai_output, tool_outputs) -> list[str] of
    # number-claim mismatches between the reply and the raw fetched data. The
    # hallucination gate (evaluate_hallucination_consistency) only runs when a
    # project registers one. None keeps the framework portable.
    hallucination_extractor = None

    # Pluggable project hook for the AQuA P6 LLM-as-a-Judge flow:
    # callable(ai_output, expected_outcome, retrieved_context) ->
    # {"score": float (0..1), "reason": str}. Invoked only when the cheap
    # deterministic (P4) and semantic (P5) layers cannot reach a verdict in
    # evaluate_expected_outcome. None keeps the framework fully offline and
    # deterministic (the pre-judge behavior: fail below the semantic
    # threshold).
    llm_judge = None

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

        def _executed(required):
            return required_tool_executed(required, executed_tools, tool_calls)

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
    def evaluate_expected_outcome(ai_output, expected_outcome, retrieved_context=None):
        """
        Layer X: Expected outcome checking (AQuA P3/P4/P5/P6 escalation).

        Deterministic exact match first (P4); falls back to JSON structural
        match, then semantic (cosine) similarity (P5). When the cheap semantic
        layer cannot reach the threshold, the AQuA P6 LLM-as-a-Judge flow
        escalates to the project-registered ``llm_judge`` hook, which audits
        groundedness/completeness against the golden reference using the
        retrieved context. Without a configured judge the case fails below
        the threshold (the previous behavior).
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

            # P6 LLM-as-a-Judge: the cheap layers could not decide, so escalate
            # to the project's judge (P3 progressive evaluation). The judge
            # returns a score that feeds straight into risk-based confidence.
            judge = AQuAEvaluators.llm_judge
            if judge is not None:
                verdict = judge(ai_output, expected_outcome, list(retrieved_context or []))
                score = verdict.get("score")
                if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
                    score = 0.0
                return {
                    "check_name": "llm_judge",
                    "status": "PASSED" if score >= LLM_JUDGE_PASS_THRESHOLD else "FAILED",
                    "score": score,
                    "reason": f"LLM judge verdict ({score:.2f}): {verdict.get('reason') or 'no rationale'}",
                }
            return {"check_name": "expected_outcome", "status": "FAILED", "score": 0.0,
                    "reason": f"Semantic similarity {similarity:.3f} below threshold {EXPECTED_OUTCOME_SEMANTIC_THRESHOLD}."}

    @staticmethod
    def evaluate_hallucination_consistency(ai_output, trace_logs):
        """
        Generic gate: numbers in the reply must match the raw fetched data.

        Uses the project-registered ``hallucination_extractor`` to enumerate
        mismatches (e.g. an earthquake count or strongest magnitude that does
        not appear in ``tool_outputs``). Skipped by run_case when no extractor
        is configured or nothing was fetched.
        """
        extractor = AQuAEvaluators.hallucination_extractor
        mismatches = extractor(ai_output, (trace_logs or {}).get("tool_outputs") or {})
        if mismatches:
            return {
                "check_name": "hallucination_check",
                "status": "FAILED",
                "score": 0.0,
                "reason": "Reply numbers contradict fetched data: " + "; ".join(mismatches),
            }
        return {"check_name": "hallucination_check", "status": "PASSED", "score": 1.0}

    @classmethod
    def compute_metrics(cls, case, ai_output, trace_logs):
        """
        Per-case bool-or-None KPI metrics for the AQuA run ledger.

        - ``intent_accurate``: the called tool names form exactly the set of
          required tools, and every required tool that declares parameters was
          called with exactly those parameter values. None when the case
          declares no tools (excluded from the rate).
        - ``hallucinated``: the reply contradicts the fetched data. None when
          nothing was fetched (nothing to compare against).
        """
        required = case.get("required_tools") or []
        required_names = {
            t.get("name", t) if isinstance(t, dict) else t
            for t in required
        }
        if not required_names:
            intent = None
        else:
            executed_tools = (trace_logs or {}).get("executed_tools") or []
            tool_calls = (trace_logs or {}).get("executed_tool_calls") or []
            names_ok = set(executed_tools) == required_names
            params_ok = all(
                required_tool_executed(t, executed_tools, tool_calls) for t in required
            )
            intent = names_ok and params_ok

        outputs = (trace_logs or {}).get("tool_outputs") or {}
        hallucinated = None
        if outputs and cls.hallucination_extractor is not None:
            hallucinated = bool(cls.hallucination_extractor(ai_output, outputs))
        return {"intent_accurate": intent, "hallucinated": hallucinated}

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

        Returns the confidence dict plus per-case KPI ``metrics`` (computed by
        :meth:`compute_metrics`).
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

        # 4. Check Expected Outcome (P4/P5, escalating to the P6 LLM judge
        # when a judge is configured and the cheap layers cannot decide).
        if "expected_outcome" in case:
            results.append(cls.evaluate_expected_outcome(
                content,
                case.get("expected_outcome"),
                (trace_logs or {}).get("retrieved_context"),
            ))

        # 5. Check Hallucination Consistency (reply numbers vs fetched data).
        # Runs on every case that fetched data, when the project registered a
        # hallucination_extractor.
        if cls.hallucination_extractor is not None and (trace_logs or {}).get("tool_outputs"):
            results.append(cls.evaluate_hallucination_consistency(ai_output, trace_logs))

        # 6. Final Risk-Based Confidence Gate
        result = cls.calculate_confidence(results, case.get("threshold", 0.90))
        result["metrics"] = cls.compute_metrics(case, ai_output, trace_logs)
        return result

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
