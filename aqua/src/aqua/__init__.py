"""AQuA: a portable AI quality engineering framework.

The framework provides the generic evaluation toolkit for golden-anchor
testing (evaluators, run KPIs, LLM-as-a-judge, a pytest plugin and reporting
CLI). It is intentionally project-agnostic: host projects supply their own
assistant (the system under test), golden-anchor cases, and a
hallucination-extractor via pytest fixtures.

See the README for the host-project integration contract.
"""

from aqua.config import JudgeConfig, judge_config_from_env
from aqua.evaluation import AQuAEvaluators
from aqua.judge import build_llm_judge
from aqua.reporting import AQuARunLedger

__all__ = [
    "AQuAEvaluators",
    "AQuARunLedger",
    "JudgeConfig",
    "build_llm_judge",
    "judge_config_from_env",
]
