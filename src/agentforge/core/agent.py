"""ReAct agent — the loop.

    while step < max_steps:
        thought, action, action_input = LLM(question + scratchpad)
        if final_answer: return
        observation = tool(action_input)
        scratchpad += step

That is the whole idea. Everything else is just plumbing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agentforge.core.parser import parse_step
from agentforge.core.prompts import build_system_prompt, build_user_prompt, format_scratchpad
from agentforge.llm import LLM
from agentforge.tools import Tool, ToolRegistry

logger = logging.getLogger("agentforge")


@dataclass
class Step:
    thought: str
    tool: str | None
    action_input: str | None
    observation: str | None
    elapsed_ms: float


@dataclass
class AgentResult:
    question: str
    final_answer: str
    steps: list[Step]
    n_steps: int
    success: bool
    latency_ms: float
    extras: dict[str, Any] = field(default_factory=dict)


class Agent:
    """ReAct agent with a tool registry and a configurable LLM."""

    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | ToolRegistry,
        *,
        max_steps: int = 6,
        stop: tuple[str, ...] = ("\nObservation:",),
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.max_steps = max_steps
        self.stop = stop
        self.verbose = verbose

    @classmethod
    def from_defaults(
        cls,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        *,
        tools: list[Tool] | None = None,
        max_steps: int = 6,
        verbose: bool = False,
        quantize: str | None = None,
        device_map: str | dict | None = "auto",
    ) -> Agent:
        if quantize:
            from agentforge.llm import QuantizedHFLLM

            llm = QuantizedHFLLM(model_id, method=quantize, device_map=device_map)
        else:
            from agentforge.llm import HFLLM

            llm = HFLLM(model_id, device_map=device_map)

        return cls(llm=llm, tools=tools or [], max_steps=max_steps, verbose=verbose)

    # ------------------------------------------------------------------ public

    def run(
        self,
        question: str,
        *,
        max_steps: int | None = None,
        max_new_tokens: int = 256,
    ) -> AgentResult:
        max_steps = max_steps or self.max_steps
        sys_prompt = build_system_prompt(list(self.tools))
        steps: list[Step] = []
        t0 = time.perf_counter()

        for _ in range(max_steps):
            scratchpad = format_scratchpad(steps)
            user_prompt = build_user_prompt(question, scratchpad)
            full_prompt = f"{sys_prompt}\n\n{user_prompt}"

            llm_t0 = time.perf_counter()
            raw = self.llm.generate(
                full_prompt,
                max_new_tokens=max_new_tokens,
                stop=list(self.stop),
            )
            elapsed = (time.perf_counter() - llm_t0) * 1000

            parsed = parse_step(raw)
            if self.verbose:
                logger.info("step parse: %s", parsed)

            if parsed.is_final:
                steps.append(
                    Step(
                        thought=parsed.thought,
                        tool=None,
                        action_input=None,
                        observation=None,
                        elapsed_ms=elapsed,
                    )
                )
                return _success(question, parsed.final_answer or "", steps, t0)

            # Need a tool to continue. If the LLM didn't emit one, treat the
            # raw response as the final answer (graceful degradation).
            if not parsed.tool:
                return _success(question, raw.strip(), steps, t0, success=False)

            tool = self.tools.get(parsed.tool)
            if tool is None:
                observation = (
                    f"Error: unknown tool '{parsed.tool}'. "
                    f"Available: {', '.join(t.name for t in self.tools)}"
                )
            else:
                try:
                    observation = tool.run(parsed.action_input or "")
                except Exception as e:
                    observation = f"Error running {parsed.tool}: {type(e).__name__}: {e}"

            steps.append(
                Step(
                    thought=parsed.thought,
                    tool=parsed.tool,
                    action_input=parsed.action_input,
                    observation=observation,
                    elapsed_ms=elapsed,
                )
            )

        # Out of steps.
        return _success(
            question,
            steps[-1].observation if steps and steps[-1].observation else "",
            steps,
            t0,
            success=False,
        )


def _success(
    question: str,
    answer: str,
    steps: list[Step],
    t0: float,
    *,
    success: bool = True,
) -> AgentResult:
    latency_ms = (time.perf_counter() - t0) * 1000
    return AgentResult(
        question=question,
        final_answer=answer,
        steps=steps,
        n_steps=len(steps),
        success=success,
        latency_ms=round(latency_ms, 2),
    )


__all__ = ["Agent", "AgentResult", "Step"]
