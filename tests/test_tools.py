"""Unit tests for the tool registry and built-in tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentforge.tools import Calculator, PythonREPL, ToolRegistry
from agentforge.tools.sql import SQLTool


def test_registry_lookup_case_insensitive() -> None:
    reg = ToolRegistry([Calculator(), PythonREPL()])
    assert reg.get("calculator") is not None
    assert reg.get("Calculator") is not None
    assert reg.get("CALCULATOR") is not None
    assert reg.get("nope") is None
    assert len(reg) == 2
    assert "python_repl" in reg


def test_calculator_basic() -> None:
    c = Calculator()
    assert c.run("47 * 1337") == "62839"
    assert c.run("2 ** 10") == "1024"
    assert c.run("sqrt(16)") == "4"


def test_calculator_constants_and_funcs() -> None:
    c = Calculator()
    assert c.run("round(pi, 4)").startswith("3.141")
    assert c.run("max(1, 2, 3)") == "3"


def test_calculator_rejects_unknown_name() -> None:
    out = Calculator().run("__import__('os')")
    assert out.startswith("Error")


def test_calculator_rejects_attribute_access() -> None:
    out = Calculator().run("(1).real")
    assert out.startswith("Error")


def test_python_repl_basic() -> None:
    r = PythonREPL()
    out = r.run("[i*i for i in range(4)]")
    assert "[0, 1, 4, 9]" in out


def test_python_repl_blocks_imports() -> None:
    r = PythonREPL()
    out = r.run("import os")
    assert "Error" in out
    assert "import" in out.lower()


def test_python_repl_blocks_dunder_access() -> None:
    r = PythonREPL()
    out = r.run("().__class__")
    assert "Error" in out


def test_python_repl_print_capture() -> None:
    r = PythonREPL()
    out = r.run("print('hi'); 21*2")
    assert "hi" in out
    assert "42" in out


def test_sql_rejects_writes(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "demo.db"
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        cx.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])

    sql = SQLTool(db)
    assert "Error" in sql.run("DELETE FROM t")
    assert "Error" in sql.run("INSERT INTO t VALUES (3, 'c')")


def test_sql_select_returns_rows(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "demo.db"
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        cx.executemany("INSERT INTO t VALUES (?, ?)", [(1, "alice"), (2, "bob")])

    sql = SQLTool(db)
    out = sql.run("SELECT name FROM t ORDER BY id")
    assert "alice" in out
    assert "bob" in out


def test_sql_missing_db_raises() -> None:
    with pytest.raises(FileNotFoundError):
        SQLTool("/no/such/file.db")
