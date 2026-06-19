"""Tool protocol and registry.

A tool is *anything* with three things: a name (so the LLM can request it), a
description (so the LLM knows what it does), and a ``run(input_str) -> str``
method (so the orchestrator can call it). We avoid JSON schemas on purpose —
small open models often emit malformed JSON and the cost/benefit isn't there.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    def run(self, input_str: str) -> str: ...


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if not getattr(tool, "name", None):
            raise ValueError(f"Tool has no `.name`: {tool!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        # Lenient lookup so the LLM can write "Calculator", "calculator", or "calc".
        if name in self._tools:
            return self._tools[name]
        lc = name.strip().lower()
        for k, v in self._tools.items():
            if k.lower() == lc:
                return v
        return None

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    @property
    def names(self) -> list[str]:
        return list(self._tools)
