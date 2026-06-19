"""Unit tests for memory backends."""

from __future__ import annotations

from pathlib import Path

from agentforge.memory import ConversationMemory, PersistentMemory


def test_conversation_memory_window() -> None:
    m = ConversationMemory(max_messages=3)
    m.add("user", "a")
    m.add("assistant", "b")
    m.add("user", "c")
    m.add("assistant", "d")
    contents = [msg.content for msg in m.get()]
    assert contents == ["b", "c", "d"]


def test_conversation_memory_clear() -> None:
    m = ConversationMemory()
    m.add("user", "x")
    m.clear()
    assert m.get() == []
    assert len(m) == 0


def test_persistent_memory_roundtrip(tmp_path: Path) -> None:
    p = PersistentMemory(tmp_path / "mem.db", session="s1")
    p.add("user", "hello", source="cli")
    p.add("assistant", "hi")
    msgs = p.get()
    assert [m.content for m in msgs] == ["hello", "hi"]
    assert msgs[0].extras == {"source": "cli"}


def test_persistent_memory_session_isolation(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    a = PersistentMemory(db, session="alice")
    b = PersistentMemory(db, session="bob")
    a.add("user", "alice msg")
    b.add("user", "bob msg")
    assert [m.content for m in a.get()] == ["alice msg"]
    assert [m.content for m in b.get()] == ["bob msg"]


def test_persistent_memory_clear_scoped_to_session(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    a = PersistentMemory(db, session="alice")
    b = PersistentMemory(db, session="bob")
    a.add("user", "x")
    b.add("user", "y")
    a.clear()
    assert len(a) == 0
    assert len(b) == 1
