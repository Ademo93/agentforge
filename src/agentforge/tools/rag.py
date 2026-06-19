"""RAG tool — wraps a RAGforge Pipeline as an agent tool.

This is the cross-promo with the sister project. The tool exposes a single
``run(query)`` method that returns the top-k retrieved passages formatted for
the agent to read. No generation here — that's the LLM's job inside the
ReAct loop.
"""

from __future__ import annotations

from typing import Any


class RAGTool:
    name = "rag"
    description = (
        "Search the knowledge base for passages relevant to the query. "
        "Returns the top results with their source. Use this for company "
        "documents, manuals, FAQs, etc."
    )

    def __init__(self, pipeline: Any, *, top_k: int = 4, max_chars_per_hit: int = 600) -> None:
        if not (hasattr(pipeline, "search") or hasattr(pipeline, "retrieve")):
            raise TypeError("pipeline must be a ragforge.Pipeline or expose search()/retrieve()")
        self.pipeline = pipeline
        self.top_k = top_k
        self.max_chars = max_chars_per_hit

    def run(self, input_str: str) -> str:
        query = input_str.strip()
        if not query:
            return "Error: empty query"

        try:
            if hasattr(self.pipeline, "search"):
                hits = self.pipeline.search(query, top_k=self.top_k)
            else:
                hits = self.pipeline.retrieve(query, top_k=self.top_k)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

        if not hits:
            return "(no relevant passages found)"

        lines: list[str] = []
        for i, h in enumerate(hits, 1):
            text = (h.text or "").strip().replace("\n", " ")
            if len(text) > self.max_chars:
                text = text[: self.max_chars] + "..."
            src = h.metadata.get("path", h.id) if hasattr(h, "metadata") else str(h)
            page = h.metadata.get("page") if hasattr(h, "metadata") else None
            loc = f"{src} (p.{page})" if page else src
            lines.append(f"[{i}] {loc}\n    {text}")
        return "\n".join(lines)
