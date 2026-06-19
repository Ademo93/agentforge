"""ReAct agent with calculator + Python REPL — no internet needed.

Run:
    python examples/basic_agent.py --question "Square root of (47 * 1337)?"
"""

from __future__ import annotations

import argparse

from agentforge import Agent
from agentforge.tools import Calculator, PythonREPL
from agentforge.utils import use_truststore


def main() -> None:
    use_truststore()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    agent = Agent.from_defaults(
        model_id=args.model_id,
        tools=[Calculator(), PythonREPL()],
        max_steps=args.max_steps,
        verbose=True,
    )
    result = agent.run(args.question)

    print(f"\n[final answer in {result.latency_ms:.0f} ms, {result.n_steps} steps]")
    print(result.final_answer)
    print("\nsteps:")
    for i, s in enumerate(result.steps, 1):
        if s.tool:
            print(f"  [{i}] {s.tool}({s.action_input!r}) -> {s.observation!r}")
        else:
            print(f"  [{i}] {s.thought}")


if __name__ == "__main__":
    main()
