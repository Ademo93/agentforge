"""Unit tests for the ReAct output parser."""

from __future__ import annotations

from agentforge.core.parser import parse_step


def test_parse_thought_action_input() -> None:
    text = "Thought: I should use the calculator\nAction: calculator\nAction Input: 2 + 2"
    p = parse_step(text)
    assert p.thought == "I should use the calculator"
    assert p.tool == "calculator"
    assert p.action_input == "2 + 2"
    assert not p.is_final


def test_parse_final_answer() -> None:
    text = "Thought: I have the answer\nFinal Answer: 4"
    p = parse_step(text)
    assert p.is_final
    assert p.final_answer == "4"


def test_parse_is_case_insensitive_and_tolerates_extra_lines() -> None:
    text = "thought: reason\naction: WebSearch\naction input:   what is RAG?\nObservation: ..."
    p = parse_step(text)
    assert p.tool == "WebSearch"
    assert p.action_input == "what is RAG?"


def test_parse_trims_trailing_observation_block() -> None:
    text = (
        "Thought: a\nAction: calculator\nAction Input: 1+1\nObservation: 2\n"
        "Thought: b\nAction: calculator\nAction Input: 2+2"
    )
    p = parse_step(text)
    # The parser grabs the *first* action by design; the trailing block is
    # ignored — the LLM should only emit one step at a time.
    assert p.tool == "calculator"
    assert p.action_input == "1+1"


def test_parse_final_answer_strips_trailing_garbage() -> None:
    text = "Final Answer: 42\nThought: should not appear"
    p = parse_step(text)
    assert p.final_answer == "42"
