"""ReAct agent that retrieves from a RAGforge index.

Demonstrates the full open-source stack:
    turboquant-ml (optional, quantize the LLM)
        + ragforge-ml (retrieval over your docs)
        + agentforge-ml (this — tool-using agent)

Run (after `pip install "agentforge-ml[rag]"`):
    python examples/rag_agent.py --docs ./data --question "How do I rotate an API key?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentforge import Agent
from agentforge.tools import Calculator, RAGTool
from agentforge.utils import use_truststore


def main() -> None:
    use_truststore()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--docs", type=Path, default=Path("data"))
    parser.add_argument("--collection", default="agentforge-demo")
    parser.add_argument("--store-path", type=Path, default=Path("qdrant_storage"))
    parser.add_argument("--question", required=True)
    parser.add_argument("--quantize", default=None, help="Optional turboquant-ml method (e.g. bnb-nf4)")
    args = parser.parse_args()

    from ragforge import Pipeline

    rag = Pipeline.from_defaults(
        collection=args.collection,
        store_path=args.store_path,
    )
    n = rag.ingest([args.docs])
    print(f"indexed {n} chunks")

    agent = Agent.from_defaults(
        model_id=args.model_id,
        tools=[RAGTool(rag), Calculator()],
        max_steps=6,
        quantize=args.quantize,
        verbose=True,
    )
    result = agent.run(args.question)

    print(f"\n[final answer in {result.latency_ms:.0f} ms, {result.n_steps} steps]")
    print(result.final_answer)


if __name__ == "__main__":
    main()
