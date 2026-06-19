"""LLM protocol — anything that maps a prompt to text and honors stop strings."""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str: ...
