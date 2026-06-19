"""ReAct output parser.

The LLM emits free-form text shaped roughly like:

    Thought: ...
    Action: tool_name
    Action Input: ...

or:

    Thought: ...
    Final Answer: ...

This parser is intentionally **forgiving** — small open models often forget
whitespace, mix cases, or fail to close the action input. We grab the most
recent plausible step from the buffer and stop at the first unambiguous
boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FINAL = re.compile(r"final\s*answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
_THOUGHT = re.compile(
    r"thought\s*:\s*(.*?)(?=\n\s*(?:action|final\s*answer)\s*:|$)", re.IGNORECASE | re.DOTALL
)
_ACTION = re.compile(r"action\s*:\s*([^\n]+)", re.IGNORECASE)
_INPUT = re.compile(
    r"action\s*input\s*:\s*(.*?)(?=\n\s*(?:observation|thought|action|final\s*answer)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ParsedStep:
    thought: str = ""
    tool: str | None = None
    action_input: str | None = None
    final_answer: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


def parse_step(text: str) -> ParsedStep:
    """Pull the next Thought/Action/Action Input *or* Final Answer out of ``text``."""
    text = text.strip()

    final = _FINAL.search(text)
    if final:
        # If the model emitted a Final Answer, anything before it is the thought.
        thought = _extract_thought(text[: final.start()])
        ans = final.group(1).strip()
        # Trim a possible trailing block ("Observation: ...") in case the model went past.
        ans = _trim_after(ans, ("\nThought:", "\nObservation:", "\nAction:"))
        return ParsedStep(thought=thought, final_answer=ans)

    thought = _extract_thought(text)
    action = _ACTION.search(text)
    inp = _INPUT.search(text)
    tool = action.group(1).strip() if action else None
    action_input = inp.group(1).strip() if inp else None
    if action_input is not None:
        action_input = _trim_after(action_input, ("\nThought:", "\nObservation:", "\nAction:"))
    return ParsedStep(thought=thought, tool=tool, action_input=action_input)


def _extract_thought(text: str) -> str:
    m = _THOUGHT.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def _trim_after(s: str, stops: tuple[str, ...]) -> str:
    earliest = len(s)
    lower = s.lower()
    for stop in stops:
        idx = lower.find(stop.lower())
        if idx != -1:
            earliest = min(earliest, idx)
    return s[:earliest].strip()
