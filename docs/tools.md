# Tools

Every tool has three things:

| | |
|---|---|
| `name` | Identifier the LLM uses in `Action:` |
| `description` | One-line explanation the LLM reads in the system prompt |
| `run(input_str) -> str` | Synchronous handler returning the observation |

## Built-in

### `calculator`

Safe arithmetic via an **AST whitelist**. No `eval()`. Supports `+ - * / // % **`,
parentheses, and `sqrt log exp sin cos tan abs round floor ceil max min`.
Constants `pi`, `e`. Rejects attribute access, imports, and any unknown name.

### `python_repl`

Run short Python in a sandbox. The sandbox blocks:
- Imports
- Attribute access to dunders (`__class__`, `__globals__`, …)
- `eval`, `exec`, `compile`, `open`, `__import__`, etc.

Supports comprehensions, list/dict/string ops, and the safe builtins
(`len, range, sorted, sum, ...`). Captures `print()` output and the value of
the last expression.

### `web_search`

DuckDuckGo, no API key. Returns the top-k results as `title / url / snippet`.

```python
WebSearch(top_k=5, region="wt-wt")
```

### `sql`

Read-only sqlite. Connects with `?mode=ro` so the LLM cannot mutate the
database. Rejects everything except `SELECT`, `PRAGMA`, `WITH`. Returns rows
as a compact text table.

```python
SQLTool("./mydb.sqlite", max_rows=50)
```

### `rag`

Wraps a [`ragforge`](https://github.com/Ademo93/ragforge) Pipeline. Calls
`pipeline.search()` (retrieve + rerank) and formats the top hits for the agent.

```python
from ragforge import Pipeline
from agentforge.tools import RAGTool

rag = Pipeline.from_defaults(model_id=None)  # no LLM in the pipeline — the agent has one
rag.ingest(["docs/"])

tool = RAGTool(rag, top_k=4)
```

## Writing your own

A tool is anything with `name`, `description`, and `.run(str) -> str`:

```python
class WeatherTool:
    name = "weather"
    description = "Get the current weather for a city. Input: city name."

    def run(self, input_str: str) -> str:
        return get_weather(input_str.strip())

agent = Agent.from_defaults(model_id="...", tools=[WeatherTool()])
```

That is it. No JSON schema, no decorators, no callbacks.

## Registry

`ToolRegistry` is dict-like and **case-insensitive on lookup**, so the LLM
can write "Calculator", "calculator", or "CALCULATOR" and the right tool
gets called:

```python
from agentforge.tools import ToolRegistry, Calculator

reg = ToolRegistry([Calculator()])
reg.get("Calculator")  # works
reg.get("calc")        # None — only the canonical name and aliases match
```
