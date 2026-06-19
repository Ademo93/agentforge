"""Unit tests for the ReAct agent loop, using a scripted LLM."""

from __future__ import annotations

from agentforge.core.agent import Agent
from agentforge.tools import Calculator


def test_agent_runs_calculator_and_returns_final(scripted_llm) -> None:
    llm = scripted_llm(
        [
            "Thought: Use the calculator\nAction: calculator\nAction Input: 47 * 1337",
            "Thought: Done\nFinal Answer: 62839",
        ]
    )
    agent = Agent(llm=llm, tools=[Calculator()], max_steps=4)
    result = agent.run("What is 47 * 1337?")
    assert result.success
    assert result.n_steps == 2
    assert "62839" in result.final_answer
    assert result.steps[0].tool == "calculator"
    assert "62839" in (result.steps[0].observation or "")


def test_agent_handles_unknown_tool(scripted_llm) -> None:
    llm = scripted_llm(
        [
            "Thought: try the wrong tool\nAction: nosuchtool\nAction Input: hi",
            "Thought: ok\nFinal Answer: gave up",
        ]
    )
    agent = Agent(llm=llm, tools=[Calculator()], max_steps=4)
    result = agent.run("anything")
    assert result.success
    assert "Error" in (result.steps[0].observation or "")
    assert "nosuchtool" in (result.steps[0].observation or "")


def test_agent_short_circuits_on_immediate_final_answer(scripted_llm) -> None:
    llm = scripted_llm(
        [
            "Thought: trivial\nFinal Answer: yes",
        ]
    )
    agent = Agent(llm=llm, tools=[Calculator()], max_steps=4)
    result = agent.run("Are you sure?")
    assert result.success
    assert result.n_steps == 1
    assert result.final_answer == "yes"


def test_agent_respects_max_steps(scripted_llm) -> None:
    # Never emits Final Answer — should run out of steps and report success=False.
    looping = ["Thought: ?\nAction: calculator\nAction Input: 1+1"] * 10
    llm = scripted_llm(looping)
    agent = Agent(llm=llm, tools=[Calculator()], max_steps=3)
    result = agent.run("loop forever")
    assert not result.success
    assert result.n_steps == 3
