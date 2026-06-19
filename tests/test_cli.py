"""CLI smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from agentforge.cli import app

runner = CliRunner()


def test_help_lists_subcommands() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("ask", "eval", "serve", "tools"):
        assert cmd in res.stdout


def test_tools_subcommand_runs() -> None:
    res = runner.invoke(app, ["tools"])
    assert res.exit_code == 0
    assert "calculator" in res.stdout
