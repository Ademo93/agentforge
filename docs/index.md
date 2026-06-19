# AgentForge

> ReAct agents on open-weight LLMs — tools, memory, and an eval harness. Pairs with [ragforge-ml](https://github.com/Ademo93/ragforge) for retrieval and [turboquant-ml](https://github.com/Ademo93/turboquant) for quantized model serving.

## What it is

A small, readable ReAct agent runtime that you can run **end-to-end on your
own machine with open models, no API key required**. Each tool is one short
module behind a tiny interface (`name`, `description`, `run(input) -> str`)
and every step in the loop is auditable.

Three opinions:

1. **Open models first.** Defaults to `Qwen/Qwen2.5-3B-Instruct`. No OpenAI
   key required to try.
2. **ReAct, not magic.** The loop is a single 60-line function. Compare with
   any "agent framework" — you can read this one in a coffee break.
3. **Tools with hard limits.** Python REPL is AST-whitelisted, SQL is
   read-only, web search is rate-limited.

## Install

```bash
pip install agentforge-ml                       # core
pip install "agentforge-ml[tools]"              # + sympy + duckduckgo-search
pip install "agentforge-ml[rag]"                # + ragforge-ml
pip install "agentforge-ml[quantized]"          # + turboquant-ml NF4
pip install "agentforge-ml[serve]"              # + FastAPI
pip install "agentforge-ml[all]"                # everything
```

## 60-second tour

```python
from agentforge import Agent
from agentforge.tools import Calculator, PythonREPL

agent = Agent.from_defaults(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    tools=[Calculator(), PythonREPL()],
)

result = agent.run("What is 47 * 1337, then take its square root?")
print(result.final_answer)
for step in result.steps:
    print(f"  [{step.tool}] {step.action_input!r} -> {step.observation!r}")
```

## CLI

```bash
af ask "How many primes below 100?" --tools python_repl
af eval examples/eval_set.jsonl
af serve --tools calculator,python_repl
```

See **[ReAct](react.md)**, **[Tools](tools.md)**, **[Memory](memory.md)**,
**[Evaluation](evaluation.md)**.
