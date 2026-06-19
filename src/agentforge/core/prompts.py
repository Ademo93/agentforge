"""ReAct prompt templates.

The system prompt explains the loop format and lists the available tools with
their docstrings. We keep it short and concrete — long prompts confuse small
open models more than they help.
"""

from __future__ import annotations

REACT_SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step.

You have access to the following tools:

{tool_block}

Use this exact format, one Thought/Action/Action Input per step:

Thought: <your reasoning about what to do next>
Action: <one of: {tool_names}>
Action Input: <the input to the tool>

After each step, you will receive an Observation: with the tool's output.
Continue until you can answer, then write:

Thought: <reasoning summarizing the result>
Final Answer: <your final, concise answer>

Important rules:
- Use only one Action per step. Do not write multiple actions at once.
- If the answer is straightforward and no tool is needed, you may go directly to "Final Answer:".
- Do not invent observations — wait for the system to provide them.
"""


def build_system_prompt(tools: list) -> str:
    """Render the tool registry into the system prompt."""
    tool_block = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    tool_names = ", ".join(t.name for t in tools) if tools else "(none)"
    return REACT_SYSTEM_PROMPT.format(tool_block=tool_block, tool_names=tool_names)


def build_user_prompt(question: str, scratchpad: str = "") -> str:
    """Render the user turn: the question + the running scratchpad of prior steps."""
    if not scratchpad:
        return f"Question: {question}\n\nThought:"
    return f"Question: {question}\n\n{scratchpad}\nThought:"


def format_scratchpad(steps: list) -> str:
    """Concatenate completed steps back into the prompt for the next iteration."""
    parts = []
    for s in steps:
        parts.append(
            f"Thought: {s.thought}\n"
            f"Action: {s.tool}\n"
            f"Action Input: {s.action_input}\n"
            f"Observation: {s.observation}"
        )
    return "\n\n".join(parts)
