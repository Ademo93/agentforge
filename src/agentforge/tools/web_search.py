"""Web search via DuckDuckGo — no API key.

Returns a formatted list of the top-k results: title, url, snippet. The result
is intentionally compact (model context is expensive) and includes URLs so the
LLM can quote sources.
"""

from __future__ import annotations


class WebSearch:
    name = "web_search"
    description = (
        "Search the web. Input is a search query. Returns up to top_k results: title, url, snippet."
    )

    def __init__(self, *, top_k: int = 5, region: str = "wt-wt") -> None:
        self.top_k = top_k
        self.region = region

    def run(self, input_str: str) -> str:
        query = input_str.strip()
        if not query:
            return "Error: empty query"
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return (
                "Error: duckduckgo-search not installed. "
                'Install with `pip install "agentforge-ml[tools]"`.'
            )

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region=self.region, max_results=self.top_k))
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

        if not results:
            return "(no results)"

        out: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()
            url = r.get("href", r.get("url", "")).strip()
            body = r.get("body", "").strip()
            out.append(f"[{i}] {title}\n    {url}\n    {body[:240]}")
        return "\n".join(out)
