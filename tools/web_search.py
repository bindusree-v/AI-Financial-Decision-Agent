"""
Web search tool using Tavily API.
Supports iterative, multi-step financial research queries.
"""
from __future__ import annotations

import logging
from typing import Any

from tavily import TavilyClient

from config import config

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Wraps Tavily search for financial research.
    Returns structured results with title, url, content, and score.
    """

    def __init__(self) -> None:
        if not config.TAVILY_API_KEY:
            raise EnvironmentError("TAVILY_API_KEY is not set.")
        self.client = TavilyClient(api_key=config.TAVILY_API_KEY)
        self.max_results = config.MAX_SEARCH_RESULTS

    def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int | None = None,
        include_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a web search and return cleaned results.

        Args:
            query: The search query string.
            search_depth: "basic" or "advanced" (advanced = more thorough).
            max_results: Override default max results.
            include_domains: Restrict search to specific domains.

        Returns:
            List of dicts with keys: title, url, content, score.
        """
        n = max_results or self.max_results
        try:
            params: dict[str, Any] = {
                "query": query,
                "search_depth": search_depth,
                "max_results": n,
                "include_answer": True,
            }
            if include_domains:
                params["include_domains"] = include_domains

            response = self.client.search(**params)
            results = response.get("results", [])

            cleaned = []
            for r in results:
                cleaned.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0.0),
                    }
                )

            # Also include the synthesized answer if available
            answer = response.get("answer", "")
            if answer:
                cleaned.insert(
                    0,
                    {
                        "title": "Tavily Synthesized Answer",
                        "url": "",
                        "content": answer,
                        "score": 1.0,
                    },
                )

            logger.info("Search '%s' returned %d results.", query, len(cleaned))
            return cleaned

        except Exception as exc:
            logger.error("Web search failed for query '%s': %s", query, exc)
            return [
                {
                    "title": "Search Error",
                    "url": "",
                    "content": f"Search failed: {exc}",
                    "score": 0.0,
                }
            ]

    def format_results_as_text(self, results: list[dict[str, Any]]) -> str:
        """Convert search results list to a readable text block."""
        if not results:
            return "No results found."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r['title']}\n"
                f"    URL: {r['url']}\n"
                f"    {r['content'][:600]}"
            )
        return "\n\n".join(parts)


# Module-level singleton
_search_tool: WebSearchTool | None = None


def get_search_tool() -> WebSearchTool:
    global _search_tool
    if _search_tool is None:
        _search_tool = WebSearchTool()
    return _search_tool
