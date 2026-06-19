"""Shared fixtures — scripted fake LLMs so tests run offline."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


class ScriptedLLM:
    """Returns a pre-canned response per ``generate()`` call.

    Lets us script ReAct conversations deterministically:

        llm = ScriptedLLM([
            "Thought: Use the calculator\\nAction: calculator\\nAction Input: 2 + 2",
            "Thought: Done\\nFinal Answer: 4",
        ])
    """

    model_id = "test/scripted-llm"

    def __init__(self, responses: list[str]) -> None:
        self._responses: Iterator[str] = iter(responses)
        self.calls = 0

    def generate(self, prompt: str, **_: object) -> str:
        self.calls += 1
        try:
            return next(self._responses)
        except StopIteration as e:
            raise RuntimeError(f"ScriptedLLM ran out of responses after {self.calls} calls") from e


@pytest.fixture
def scripted_llm():
    return ScriptedLLM
