"""Agent-quality metrics.

Pure Python, no judge model. The metrics are intentionally simple so the
relationship between "the agent did X" and "the metric says Y" is auditable
from the code alone.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agentforge.core.agent import AgentResult


def task_completion(result: AgentResult, sample: dict) -> float:
    """Did the agent produce *any* final answer? (Did the loop terminate cleanly?)"""
    return 1.0 if result.success and result.final_answer.strip() else 0.0


def final_answer_match(result: AgentResult, sample: dict) -> float:
    """Does the final answer contain the ground-truth string (case-folded)?

    Substring rather than exact match because LLMs tend to wrap answers in
    natural-language framing ("The answer is 42."). For numeric questions,
    pass the bare number as ``ground_truth`` and it will be matched.
    """
    gt = sample.get("ground_truth")
    if not gt:
        return 0.0
    return 1.0 if str(gt).strip().lower() in result.final_answer.lower() else 0.0


def tool_accuracy(result: AgentResult, sample: dict) -> float:
    """Fraction of tool calls that match the expected tool(s).

    ``expected_tools`` in the sample can be:
        - a string (one tool expected at any step)
        - a list of strings (each step's expected tool, in order, with "any" wildcard)
        - missing (returns 1.0 for trivia-style samples)
    """
    expected = sample.get("expected_tools")
    if expected is None:
        return 1.0

    actual = [s.tool for s in result.steps if s.tool]
    if isinstance(expected, str):
        return 1.0 if expected in actual else 0.0

    if not isinstance(expected, list):
        return 0.0
    if not actual:
        return 0.0

    # Index-wise comparison up to the shorter sequence.
    n = min(len(actual), len(expected))
    if n == 0:
        return 0.0
    matched = sum(1 for i in range(n) if expected[i] in ("*", "any", actual[i]))
    return matched / max(len(expected), 1)


def step_efficiency(result: AgentResult, sample: dict) -> float:
    """``ground_truth_steps / actual_steps`` clipped to [0, 1].

    Encourages reaching the answer in as few steps as possible. If the sample
    omits ``ground_truth_steps``, defaults to 1 step.
    """
    if not result.success or result.n_steps == 0:
        return 0.0
    gt_steps = max(1, int(sample.get("ground_truth_steps", 1)))
    return min(1.0, gt_steps / result.n_steps)


_REGISTRY = {
    "task_completion": task_completion,
    "final_answer_match": final_answer_match,
    "tool_accuracy": tool_accuracy,
    "step_efficiency": step_efficiency,
}


def evaluate(
    samples: list[dict[str, Any]],
    results: list[AgentResult],
    *,
    metrics: Iterable[str] = (
        "task_completion",
        "final_answer_match",
        "tool_accuracy",
        "step_efficiency",
    ),
) -> dict:
    """Score a parallel list of samples + results across one or more metrics."""
    if len(samples) != len(results):
        raise ValueError(f"len(samples)={len(samples)} != len(results)={len(results)}")

    by_metric: dict[str, list[float]] = {m: [] for m in metrics}
    for sample, result in zip(samples, results, strict=True):
        for m in metrics:
            fn = _REGISTRY[m]
            by_metric[m].append(float(fn(result, sample)))

    means = {m: (sum(v) / len(v) if v else 0.0) for m, v in by_metric.items()}
    return {"n": len(samples), "means": means, "per_sample": by_metric}
