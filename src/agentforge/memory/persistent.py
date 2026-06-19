"""SQLite-backed persistent memory.

Sessions are keyed by an arbitrary string id — useful for multi-user agents
where each user has their own running history. No ORM, no migration tooling:
a single ``messages`` table is created on first use.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from agentforge.memory.base import Message


class PersistentMemory:
    def __init__(
        self, db_path: str | Path = "agentforge_memory.db", *, session: str = "default"
    ) -> None:
        self.db_path = str(db_path)
        self.session = session
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as cx:
            cx.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts REAL NOT NULL,
                    extras_json TEXT
                )
                """
            )
            cx.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_ts ON messages(session, ts)"
            )

    def add(self, role: str, content: str, **extras: object) -> None:
        with sqlite3.connect(self.db_path) as cx:
            cx.execute(
                "INSERT INTO messages (session, role, content, ts, extras_json) VALUES (?, ?, ?, ?, ?)",
                (self.session, role, content, time.time(), json.dumps(extras) if extras else None),
            )

    def get(self, *, limit: int | None = None) -> list[Message]:
        with sqlite3.connect(self.db_path) as cx:
            cx.row_factory = sqlite3.Row
            q = "SELECT role, content, ts, extras_json FROM messages WHERE session = ? ORDER BY ts ASC"
            rows = cx.execute(q, (self.session,)).fetchall()
        msgs = [
            Message(
                role=r["role"],
                content=r["content"],
                ts=r["ts"],
                extras=json.loads(r["extras_json"]) if r["extras_json"] else {},
            )
            for r in rows
        ]
        if limit:
            msgs = msgs[-limit:]
        return msgs

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as cx:
            cx.execute("DELETE FROM messages WHERE session = ?", (self.session,))

    def __len__(self) -> int:
        with sqlite3.connect(self.db_path) as cx:
            return int(
                cx.execute(
                    "SELECT COUNT(*) FROM messages WHERE session = ?", (self.session,)
                ).fetchone()[0]
            )
