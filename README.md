<h1 align="center">AgentForge</h1>

<p align="center">
  <strong>ReAct agents on open-weight LLMs — tools, memory, and an eval harness.</strong>
  <br>
  Pairs with <a href="https://github.com/Ademo93/ragforge">ragforge-ml</a> for retrieval and
  <a href="https://github.com/Ademo93/turboquant">turboquant-ml</a> for quantized model serving.
</p>

<p align="center">
  <a href="https://pypi.org/project/agentforge-ml/"><img alt="PyPI" src="https://img.shields.io/badge/pypi-agentforge--ml-blue"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://Ademo93.github.io/agentforge/"><img alt="Docs" src="https://img.shields.io/badge/docs-mkdocs--material-blue"></a>
</p>

---

## Why AgentForge?

Most "agent framework" projects use proprietary models (GPT-4, Claude) behind a
DSL of `Runnable.invoke()` chains nobody can debug. AgentForge is the opposite:
**ReAct loops on open-weight LLMs (Llama, Qwen, Mistral), with a small registry
of well-bounded tools, and an evaluation harness so you can measure whether
your agent is actually doing what you asked.**

Three opinions:

1. **Open models first.** Defaults work on `Qwen/Qwen2.5-3B-Instruct` and any
   chat-template HF model. No API key required. Plug in
   [turboquant-ml](https://github.com/Ademo93/turboquant) to serve the model
   quantized.
2. **ReAct, not magic.** The loop is a 60-line function (`agent.py:run`) that
   alternates Thought / Action / Observation steps. Easy to read, easy to debug.
3. **Tools have hard boundaries.** Python REPL runs in an AST-whitelisted
   sandbox; SQL is read-only; web search is rate-limited; RAG retrieval is
   delegated to [ragforge-ml](https://github.com/Ademo93/ragforge).

## Features

| Stage | Default |
|---|---|
| **LLM** | Any HuggingFace chat-template model. Optional `bnb-nf4` via `turboquant-ml`. |
| **Loop** | ReAct with `max_steps`, structured Thought/Action/Observation parser |
| **Tools** | `calculator`, `python` (sandboxed), `web_search` (DuckDuckGo), `sql` (read-only sqlite), `rag` (RAGforge) |
| **Memory** | In-memory conversation, persistent SQLite store |
| **Eval** | `task_completion`, `tool_accuracy`, `step_efficiency`, `final_answer_match` |
| **Serve** | FastAPI `/ask`, `/tools`, `/health` |
| **CLI** | `agentforge ask / eval / tools / serve` |

## Installation

The PyPI distribution is `agentforge-ml` (the unsuffixed `agentforge` name was
taken by an unrelated project). Python import and CLI are just `agentforge` /
`af`:

```bash
pip install agentforge-ml                       # core
pip install "agentforge-ml[tools]"              # + sympy + duckduckgo-search
pip install "agentforge-ml[rag]"                # + ragforge-ml integration
pip install "agentforge-ml[quantized]"          # + turboquant-ml NF4 path
pip install "agentforge-ml[serve]"              # + FastAPI
pip install "agentforge-ml[all]"                # everything
```

## 60-second tour

```python
from agentforge import Agent
from agentforge.tools import Calculator, WebSearch, PythonREPL

agent = Agent.from_defaults(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    tools=[Calculator(), PythonREPL(), WebSearch()],
)

result = agent.run("What is 47 * 1337, then take its square root?")
print(result.final_answer)
for step in result.steps:
    print(f"  [{step.tool}] {step.action_input!r} -> {step.observation!r}")
```

### With RAG

```python
from agentforge import Agent
from agentforge.tools import RAGTool
from ragforge import Pipeline

rag = Pipeline.from_defaults(model_id="Qwen/Qwen2.5-3B-Instruct")
rag.ingest(["docs/"])

agent = Agent.from_defaults(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    tools=[RAGTool(rag)],
)
print(agent.run("What is our company refund policy?").final_answer)
```

### CLI

```bash
af ask "What is 17 squared?" --tools calculator
af ask "Latest CVE for log4j?" --tools web_search
af eval data/eval_set.jsonl --tools calculator,python_repl
af serve --tools calculator,python_repl --port 8080
```

## ReAct loop, in a picture

```text
question -> [LLM] Thought + Action -> [Tool] Observation
            ^                                       |
            |_______________________________________|
                       up to max_steps
```

If the LLM emits `Final Answer:` the loop exits. Otherwise it loops until
`max_steps`. The parser is forgiving: it tolerates whitespace and case but
falls back to the last completed step on truncation.

## Eval harness

Built-in, pure Python, no judge model required:

| Metric | What it measures |
|---|---|
| **`task_completion`** | Did the agent produce a `Final Answer:`? |
| **`final_answer_match`** | Does the answer contain the ground-truth string (case-folded substring)? |
| **`tool_accuracy`** | Of the steps, what fraction used the expected tool? |
| **`step_efficiency`** | `ground_truth_steps / actual_steps`, clipped to [0, 1] |

```bash
af eval examples/eval_set.jsonl --tools all
```

```text
+--------------------+--------+
|  metric            |  mean  |
+--------------------+--------+
| task_completion    |  0.95  |
| final_answer_match |  0.81  |
| tool_accuracy      |  0.88  |
| step_efficiency    |  0.72  |
+--------------------+--------+
n=80  ·  p50=2.4s  ·  p95=8.1s
```

## Architecture

```
agentforge/
├── core/         # ReAct loop + parser + prompts
├── tools/        # registry, calculator, python repl, web search, sql, rag
├── memory/       # conversation, persistent sqlite
├── llm/          # HuggingFace causal LM wrapper
├── eval/         # 4 metrics + orchestrator
├── serve/        # FastAPI app
└── cli.py        # af / agentforge
```

Every stage is a small module behind a small interface (`LLM`, `Tool`,
`Memory`) — swap any of them in two lines.

## Roadmap

- [x] ReAct loop with structured parsing
- [x] Tool protocol + registry
- [x] 5 built-in tools (calculator, python, web, sql, rag)
- [x] Persistent SQLite memory
- [x] Eval: task completion, final-answer match, tool accuracy, step efficiency
- [x] FastAPI server + Typer CLI
- [x] turboquant-ml integration (NF4 / GPTQ / AWQ models)
- [ ] Plan-and-execute pattern alongside ReAct
- [ ] Streaming step output in `/ask`
- [ ] Tool-use chat templates (Qwen tool format, Llama-3 tool format)
- [ ] Multi-agent coordination

## License

[MIT](LICENSE).
