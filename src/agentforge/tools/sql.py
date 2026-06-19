"""SQL tool — read-only sqlite queries.

Connects to a sqlite file in read-only mode (``file:...?mode=ro``) so the
agent cannot mutate the DB even if the LLM asks. Returns rows as a small
text table.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLTool:
    name = "sql"
    description = (
        "Run a SELECT query against the connected SQLite database (read-only). "
        "Input is a single SQL statement. "
        "The schema can be inspected with: SELECT name FROM sqlite_master WHERE type='table';"
    )

    def __init__(self, db_path: str | Path, *, max_rows: int = 50) -> None:
        self.db_path = str(db_path)
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")
        self.max_rows = max_rows

    def run(self, input_str: str) -> str:
        sql = input_str.strip().rstrip(";")
        if not sql:
            return "Error: empty query"

        if not _is_safe_statement(sql):
            return "Error: only SELECT and PRAGMA statements are allowed."

        uri = f"file:{self.db_path}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as cx:
                cur = cx.execute(sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.Error as e:
            return f"Error: {type(e).__name__}: {e}"

        return _format(cols, rows, max_rows=self.max_rows)


def _is_safe_statement(sql: str) -> bool:
    head = sql.lstrip().split(None, 1)[0].lower() if sql.strip() else ""
    return head in {"select", "pragma", "with"}


def _format(cols: list[str], rows: list, *, max_rows: int) -> str:
    if not cols:
        return "(query produced no result set)"
    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]
    if not rows:
        return f"columns: {cols}\n(0 rows)"
    header = " | ".join(cols)
    body = "\n".join(" | ".join(str(c) for c in r) for r in rows)
    footer = f"\n... ({len(rows)} rows; truncated)" if truncated else f"\n({len(rows)} rows)"
    return f"{header}\n{body}{footer}"
