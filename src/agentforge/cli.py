"""Command-line interface — ``agentforge`` / ``af``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="agentforge",
    help="ReAct agents on open-weight LLMs with tools and an eval harness.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

_TOOL_FACTORY = {
    "calculator": lambda: _import("agentforge.tools.Calculator")(),
    "python_repl": lambda: _import("agentforge.tools.PythonREPL")(),
    "web_search": lambda: _import("agentforge.tools.WebSearch")(),
}


def _import(dotted: str):
    mod_name, cls_name = dotted.rsplit(".", 1)
    import importlib

    return getattr(importlib.import_module(mod_name), cls_name)


def _build_tools(spec: str):
    if spec.lower() in ("all", "*"):
        spec = ",".join(_TOOL_FACTORY)
    tools = []
    for name in (x.strip() for x in spec.split(",")):
        if not name:
            continue
        if name not in _TOOL_FACTORY:
            console.print(f"[red]unknown tool: {name}[/]  (available: {', '.join(_TOOL_FACTORY)})")
            raise typer.Exit(2)
        tools.append(_TOOL_FACTORY[name]())
    return tools


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    model_id: Annotated[str, typer.Option()] = "Qwen/Qwen2.5-3B-Instruct",
    tools: Annotated[str, typer.Option(help="Comma-separated or 'all'")] = "calculator,python_repl",
    max_steps: Annotated[int, typer.Option()] = 6,
    max_new_tokens: Annotated[int, typer.Option()] = 256,
    quantize: Annotated[str | None, typer.Option(help="Optional turboquant-ml method")] = None,
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run the ReAct agent on one question."""
    from agentforge import Agent

    console.print(Panel.fit(f"[bold teal]agentforge ask[/]  [dim]{question}[/]"))
    tool_list = _build_tools(tools)
    agent = Agent.from_defaults(
        model_id=model_id, tools=tool_list, max_steps=max_steps, verbose=verbose, quantize=quantize
    )
    result = agent.run(question, max_new_tokens=max_new_tokens)

    if verbose or not result.success:
        table = Table(title="Steps", show_header=True)
        table.add_column("#", style="dim")
        table.add_column("tool")
        table.add_column("input")
        table.add_column("observation")
        for i, s in enumerate(result.steps, 1):
            table.add_row(
                str(i), s.tool or "-", (s.action_input or "")[:60], (s.observation or "")[:80]
            )
        console.print(table)

    console.print(
        f"\n[bold]Final answer[/]  [dim]({result.latency_ms:.0f} ms, {result.n_steps} steps)[/]"
    )
    console.print(result.final_answer)


@app.command()
def eval(
    dataset: Annotated[
        Path, typer.Argument(help="JSONL with {question, ground_truth, expected_tools?}")
    ],
    model_id: Annotated[str, typer.Option()] = "Qwen/Qwen2.5-3B-Instruct",
    tools: Annotated[str, typer.Option()] = "all",
    max_steps: Annotated[int, typer.Option()] = 6,
    out: Annotated[Path | None, typer.Option()] = None,
    limit: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Run the eval harness over a JSONL dataset."""
    from agentforge import Agent
    from agentforge.eval import evaluate
    from agentforge.eval.report import EvalReport

    samples = _read_jsonl(dataset, limit=limit)
    console.print(Panel.fit(f"[bold teal]agentforge eval[/]  n={len(samples)}"))

    tool_list = _build_tools(tools)
    agent = Agent.from_defaults(model_id=model_id, tools=tool_list, max_steps=max_steps)

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
    console.print(report.as_table())
    if out:
        report.save(out)
        console.print(f"[green]ok[/]  saved {out}")


@app.command()
def serve(
    model_id: Annotated[str, typer.Option()] = "Qwen/Qwen2.5-3B-Instruct",
    tools: Annotated[str, typer.Option()] = "calculator,python_repl",
    max_steps: Annotated[int, typer.Option()] = 6,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Start the FastAPI agent server."""
    import uvicorn

    from agentforge import Agent
    from agentforge.serve import build_app

    tool_list = _build_tools(tools)
    agent = Agent.from_defaults(model_id=model_id, tools=tool_list, max_steps=max_steps)
    app_ = build_app(agent)
    uvicorn.run(app_, host=host, port=port, log_level="info")


@app.command(name="tools")
def list_tools_cmd() -> None:
    """List the built-in tools."""
    table = Table(title="Built-in tools")
    table.add_column("name")
    table.add_column("description")
    for name, factory in _TOOL_FACTORY.items():
        t = factory()
        table.add_row(name, t.description)
    console.print(table)


def _read_jsonl(path: Path, *, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
            if limit and len(rows) >= limit:
                break
    return rows


if __name__ == "__main__":
    app()
