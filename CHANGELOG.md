# Changelog

## [Unreleased]

## [0.1.0] — 2026-06-19 — first release

### Added
- `Agent` ReAct loop with structured Thought / Action / Observation parser.
- `ToolRegistry` + 5 built-in tools: `calculator` (AST whitelist),
  `python_repl` (sandboxed exec), `web_search` (DuckDuckGo), `sql` (read-only
  sqlite), `rag` (wraps a `ragforge` Pipeline).
- LLM backends: `HFLLM` and `QuantizedHFLLM` (via `turboquant-ml` for NF4/GPTQ/AWQ).
- Memory backends: `ConversationMemory` (in-process FIFO) and
  `PersistentMemory` (SQLite, session-scoped).
- Eval harness: `task_completion`, `final_answer_match`, `tool_accuracy`,
  `step_efficiency` with a CLI orchestrator.
- FastAPI server with `/health`, `/tools`, `/ask`.
- Typer CLI: `af ask / eval / serve / tools`.
- pytest suite (offline, scripted LLM fixture).
- CI on Python 3.10-3.12, MkDocs Material docs site.
