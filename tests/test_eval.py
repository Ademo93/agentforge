"""Unit tests for the eval metrics."""

from __future__ import annotations

from agentforge.core.agent import AgentResult, Step
from agentforge.eval import (
    evaluate,
    final_answer_match,
    step_efficiency,
    task_completion,
    tool_accuracy,
)


def _result(final: str, *steps, success: bool = True) -> AgentResult:
    return AgentResult(
        question="?",
        final_answer=final,
        steps=list(steps),
        n_steps=len(steps),
        success=success,
        latency_ms=1.0,
    )


def _step(tool: str | None = None) -> Step:
    return Step(thought="", tool=tool, action_input=None, observation=None, elapsed_ms=0.0)


def test_task_completion() -> None:
    assert task_completion(_result("yes", _step()), {}) == 1.0
    assert task_completion(_result("", _step(), success=False), {}) == 0.0


def test_final_answer_match_substring() -> None:
    r = _result("The answer is 42 because...", _step())
    assert final_answer_match(r, {"ground_truth": "42"}) == 1.0
    assert final_answer_match(r, {"ground_truth": "99"}) == 0.0
    assert final_answer_match(r, {}) == 0.0


def test_tool_accuracy_string_expectation() -> None:
    r = _result("ok", _step("calculator"), _step("python_repl"))
    assert tool_accuracy(r, {"expected_tools": "calculator"}) == 1.0
    assert tool_accuracy(r, {"expected_tools": "web_search"}) == 0.0


def test_tool_accuracy_list_expectation() -> None:
    r = _result("ok", _step("calculator"), _step("python_repl"))
    assert tool_accuracy(r, {"expected_tools": ["calculator", "python_repl"]}) == 1.0
    assert tool_accuracy(r, {"expected_tools": ["calculator", "*"]}) == 1.0
    assert tool_accuracy(r, {"expected_tools": ["web_search", "python_repl"]}) == 0.5


def test_tool_accuracy_no_expectation_is_neutral() -> None:
    r = _result("ok", _step("calculator"))
    assert tool_accuracy(r, {}) == 1.0


def test_step_efficiency() -> None:
    r = _result("ok", _step(), _step())
    assert step_efficiency(r, {"ground_truth_steps": 2}) == 1.0
    assert step_efficiency(r, {"ground_truth_steps": 1}) == 0.5
    assert step_efficiency(r, {}) == 0.5


def test_evaluate_aggregates_correctly() -> None:
    samples = [
        {
            "question": "q1",
            "ground_truth": "42",
            "expected_tools": "calculator",
            "ground_truth_steps": 2,
        },
        {"question": "q2", "ground_truth": "nope"},
    ]
    results = [
        _result("the answer is 42", _step("calculator"), _step("calculator")),
        _result("hello there"),  # no steps, success
    ]
    out = evaluate(samples, results)
    assert out["n"] == 2
    assert set(out["means"]) == {
        "task_completion",
        "final_answer_match",
        "tool_accuracy",
        "step_efficiency",
    }
    assert out["means"]["final_answer_match"] == 0.5
