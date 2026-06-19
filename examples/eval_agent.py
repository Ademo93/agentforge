"""Evaluate an agent over a JSONL dataset and emit a JSON report.

Dataset format (one JSON per line):
    {"question": "What is 47 * 1337?", "ground_truth": "62839",
     "expected_tools": "calculator", "ground_truth_steps": 2}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentforge import Agent
from agentforge.eval import evaluate
from agentforge.eval.report import EvalReport
from agentforge.tools import Calculator, PythonREPL
from agentforge.utils import use_truststore


def main() -> None:
    use_truststore()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/agent_eval.json"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples: list[dict] = []
    with args.dataset.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            samples.append(json.loads(s))
            if args.limit and len(samples) >= args.limit:
                break

    agent = Agent.from_defaults(
        model_id=args.model_id,
        tools=[Calculator(), PythonREPL()],
        max_steps=6,
    )

    results = []
    latencies: list[float] = []
    for s in samples:
        t0 = time.perf_counter()
        r = agent.run(s["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append(r)

    res = evaluate(samples, results)
    report = EvalReport(
        n=res["n"], means=res["means"], per_sample=res["per_sample"], latencies_ms=latencies
    )
    print(report.as_table())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
