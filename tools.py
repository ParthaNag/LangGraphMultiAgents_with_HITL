"""
tools.py — External Tool Wrappers
-----------------------------------
Wraps Tavily web search and a mock publish function.
Keeping tools separate from agents makes them easy to swap out
(e.g. replace Tavily with a different search API without touching agent logic).

In production, replace `publish_to_platform` with your actual CMS/API call.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def web_search(query: str, max_results: int = 3) -> list[str]:
    """
    Runs a single search query via Tavily and returns a list of result snippets.
    Called once per query inside the web_search_node in agents.py.
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query=query, max_results=max_results)
        return [r["content"] for r in response.get("results", [])]
    except Exception as e:
        # Graceful fallback so the pipeline doesn't crash during development
        print(f"[web_search] Warning: Tavily search failed for '{query}': {e}")
        return [f"[Mock result for: {query}] — Add your TAVILY_API_KEY to .env"]


def publish_to_platform(content: str) -> dict:
    """
    Mock publish function. Replace this with your actual publishing logic:
    - POST to a CMS REST API
    - Write to a database
    - Send to a Slack channel
    - Push to a GitHub repo
    Returns a dict with publish metadata.
    """
    print("\n[publisher] Content published successfully!")
    print("-" * 40)
    print(content[:300] + "..." if len(content) > 300 else content)
    print("-" * 40)
    return {
        "published": True,
        "url": "https://example.com/posts/new-post",
        "timestamp": "2026-04-09T00:00:00Z"
    }


def parse_json_list(text: str) -> list[str]:
    """
    Safely parses a JSON list from LLM output.
    LLMs sometimes wrap JSON in markdown code fences — this handles that.
    Falls back to splitting by newline if JSON parsing fails.
    """
    import json
    import re

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(item) for item in result]
        # Sometimes LLM returns {"queries": [...]} — unwrap it
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return [str(i) for i in v]
    except json.JSONDecodeError:
        pass

    # Last resort: split by newlines and strip numbering/bullets
    lines = [re.sub(r"^[\d\.\-\*\s]+", "", line).strip() for line in text.splitlines()]
    return [l for l in lines if l]
