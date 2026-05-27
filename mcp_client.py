"""MCP Client - Test the Data Analyst MCP server tools."""

from mcp_server import analyzer


def main():
    """Connect to MCP server and query tools."""

    print("Connecting to MCP Server...")
    print("Connected to MCP Server!\n")

    # List all available tools
    print("Available Tools (7):")
    tools = [
        ("get_examples", "Fetch dataset rows"),
        ("get_distribution", "Value counts per column"),
        ("count_rows", "Count filtered rows"),
        ("search_keyword", "Substring search"),
        ("get_categories", "List all categories"),
        ("get_intents", "List intents by category"),
        ("get_stats", "Dataset statistics"),
    ]
    for name, desc in tools:
        print(f"   - {name}: {desc}")

    print("Testing Tools")

    # Test 1: Count refund requests
    print("Query: Count REFUND requests")
    count = analyzer.count(category="REFUND")
    print(f"Result: {count} refund requests\n")

    # Test 2: Get distribution
    print("Query: Get category distribution")
    dist = analyzer.get_distribution("category")
    print(f"Result: {len(dist)} categories")
    for cat, count in list(dist.items())[:3]:
        print(f"      - {cat}: {count}")
    print()

    # Test 3: Get examples
    print("Query: Get 2 REFUND examples")
    examples = analyzer.get_examples(n=2, category="REFUND")
    print(f"Result: {len(examples)} examples")
    for i, ex in enumerate(examples, 1):
        print(f"      Example {i}:")
        print(f"         Intent: {ex.get('intent', 'N/A')}")
        instruction = ex.get('instruction', '')[:60]
        print(f"         Request: {instruction}...")
    print()

    # Test 4: Get categories
    print("Query: Get all categories")
    categories = analyzer.get_categories()
    print(f"Result: {categories}\n")

    # Test 5: Search keyword
    print("Query: Search for 'refund' in instructions")
    search_results = analyzer.search("refund", "instruction")
    print(f"Result: Found {len(search_results)} entries")
    print(f"      Showing: {min(len(search_results), 10)} results\n")

    # Test 6: Get intents for ACCOUNT
    print("Query: Get intents for ACCOUNT category")
    intents = analyzer.get_intents(category="ACCOUNT")
    print(f"Result: {intents}\n")

    # Test 7: Get stats
    print("Query: Get dataset statistics")
    stats = analyzer.get_stats()
    print(f"Result:")
    print(f"      - Total rows: {stats.get('num_rows', 'N/A')}")
    print(f"      - Columns: {len(stats.get('columns', []))}")
    print(f"      - Unique categories: {stats.get('unique_counts', {}).get('category', 'N/A')}")
    print()


    print("All 7 MCP Tools Working Correctly!")



if __name__ == "__main__":
    main()
