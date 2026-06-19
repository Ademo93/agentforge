"""In-memory conversation history with a rolling window."""

from __future__ import annotations

from collections import deque

from agentforge.memory.base import Message


class ConversationMemory:
    """FIFO window of the last ``max_messages`` messages."""

    def __init__(self, max_messages: int = 64) -> None:
        self.max_messages = max_messages
        self._buf: deque[Message] = deque(maxlen=max_messages)

    def add(self, role: str, content: str, **extras: object) -> None:
        self._buf.append(Message(role=role, content=content, extras=dict(extras)))

    def get(self, *, limit: int | None = None) -> list[Message]:
        msgs = list(self._buf)
        if limit:
            msgs = msgs[-limit:]
        return msgs

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
