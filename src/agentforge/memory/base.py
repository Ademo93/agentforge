"""Memory protocol + Message type."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Message:
    role: str
    content: str
    ts: float = field(default_factory=time.time)
    extras: dict = field(default_factory=dict)


class Memory(Protocol):
    def add(self, role: str, content: str, **extras: object) -> None: ...

    def get(self, *, limit: int | None = None) -> list[Message]: ...

    def clear(self) -> None: ...
