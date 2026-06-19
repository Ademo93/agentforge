# Contributing

```bash
git clone https://github.com/Ademo93/agentforge
cd agentforge
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,serve,tools,eval]"
pytest
```

## Style

- Ruff for lint and format: `ruff check . && ruff format .`
- Defer heavy imports inside functions (`transformers`, `sentence-transformers`,
  `fastapi`, `duckduckgo_search`) so unit tests stay fast.
- Each stage (core, tools, memory, llm, eval, serve) is its own subpackage with
  a tiny public surface. Keep that boundary.

## Adding a tool

1. Drop `src/agentforge/tools/your_tool.py` with a class exposing `name`,
   `description`, `run(input_str) -> str`.
2. Re-export it from `tools/__init__.py`.
3. Write a unit test that calls `tool.run(...)` with a known input.
4. Add a line to the table in `docs/tools.md`.

If the tool has side effects (network, disk, DB), the docstring **must** state
its constraints (rate limit, sandbox, read-only).

## Adding a metric

1. Add to `src/agentforge/eval/metrics.py` with the signature
   `metric(result: AgentResult, sample: dict) -> float`.
2. Register it in `_REGISTRY`.
3. Add a unit test in `tests/test_eval.py`.
4. Mention it in `docs/evaluation.md`.
