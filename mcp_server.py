"""FastMCP server exposing dataset analysis tools for Claude and other AI clients."""

import os
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from data_analysis_tools import DatasetAnalyzer

load_dotenv()

# Create FastMCP server
mcp = FastMCP("data-analyst")
analyzer = DatasetAnalyzer()


@mcp.tool()
def get_examples(
    n: int = 5,
    offset: int = 0,
    category: Optional[str] = None,
    intent: Optional[str] = None
) -> list[dict]:
    """Return up to n examples from the dataset.

    Args:
        n: Maximum number of examples to return (default 5).
        offset: Number of examples to skip for pagination (default 0).
        category: Optional filter by category (e.g., 'ACCOUNT', 'REFUND').
        intent: Optional filter by intent (e.g., 'get_refund', 'cancel_order').

    Returns:
        List of dataset rows as dicts with keys: instruction, response, category, intent.
    """
    filters = {k: v for k, v in {"category": category, "intent": intent}.items() if v}
    return analyzer.get_examples(n, offset=offset, **filters)


@mcp.tool()
def get_distribution(column: str) -> dict[str, int]:
    """Return value counts for a column.

    Args:
        column: Column name ('category', 'intent', 'flags', etc.)

    Returns:
        Dict mapping each unique value to its count, sorted by frequency.
    """
    return analyzer.get_distribution(column)


@mcp.tool()
def count_rows(category: Optional[str] = None, intent: Optional[str] = None) -> int:
    """Count rows in the dataset.

    Args:
        category: Optional filter by category.
        intent: Optional filter by intent.

    Returns:
        Integer count of matching rows.
    """
    filters = {k: v for k, v in {"category": category, "intent": intent}.items() if v}
    return analyzer.count(**filters)


@mcp.tool()
def search_keyword(keyword: str, column: str = "instruction") -> dict:
    """Search for rows where a column contains a keyword.

    Args:
        keyword: Substring to search for (case-insensitive).
        column: Column to search in (default 'instruction').

    Returns:
        Dict with 'results' (list of matching rows) and 'extra' (truncation notice if any).
    """
    rows = analyzer.search(keyword, column)
    total = len(rows)
    return {
        "results": rows[:10],
        "extra": f"[{total - 10} more entries]" if total > 10 else None,
        "total": total
    }


@mcp.tool()
def get_categories() -> list[str]:
    """Get all unique categories in the dataset.

    Returns:
        Sorted list of category names.
    """
    return analyzer.get_categories()


@mcp.tool()
def get_intents(category: Optional[str] = None) -> list[str]:
    """Get all intents, optionally filtered by category.

    Args:
        category: Optional category filter.

    Returns:
        Sorted list of intent names.
    """
    return analyzer.get_intents(category)


@mcp.tool()
def get_stats() -> dict:
    """Get dataset statistics.

    Returns:
        Dict with num_rows, columns, unique_counts, avg lengths.
    """
    return analyzer.get_stats()


if __name__ == "__main__":
    print("Starting FastMCP Server: Data Analyst")
    print("Available tools (7):")
    print("   - get_examples: Fetch dataset rows")
    print("   - get_distribution: Value counts per column")
    print("   - count_rows: Count filtered rows")
    print("   - search_keyword: Substring search")
    print("   - get_categories: List all categories")
    print("   - get_intents: List intents by category")
    print("   - get_stats: Dataset statistics")
    print("\n Connect with: mcp-client connect mcp_server.py")
    mcp.run()
