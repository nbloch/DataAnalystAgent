# Bitext Data Analyst Agent

**Students**: Tomer Porat & Nathanael Bloch

A conversational AI agent for analyzing the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Available via CLI, web UI, and MCP server.

**Assignment**: From AI Model to AI Agent - Assignment 3

## Quick Start

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`: `NEBIUS_API_KEY=your_key_here`

### Three Ways to Use

**CLI** - Interactive terminal
```bash
python main.py --session my_session
```

**Web UI** - Chat interface with session management
```bash
streamlit run streamlit_app.py
# Open http://localhost:8501
```

**MCP Server** - FastMCP for Claude and AI clients
```bash
# Terminal 1: Start server
python mcp_server.py

# Terminal 2: Test with client
python mcp_client.py
```

**Example queries:**
```
What categories exist in the dataset?
How many refund requests did we get?
Show me 5 examples of SHIPPING category
Summarize how agents respond to complaints
```

## Architecture

### Agent Graph

```
User input
    │
    ▼
[classify]  ──▶  Out-of-scope  ──▶  [reject]
    │
    ├──▶  Structured    ──▶  [ReAct, 10 iterations max]
    │
    └──▶  Unstructured  ──▶  [ReAct, 20 iterations max]
```

**Query Types:**
- **Structured**: Counts, distributions, example lookups → quick, focused reasoning
- **Unstructured**: Themes, patterns, summaries → deeper analysis
- **Out-of-scope**: Unrelated questions → polite rejection

### Model Choice

**Primary Model**: `Qwen/Qwen3-235B-A22B-Instruct-2507` (Nebius Token Factory)

**Justification**:
- **Instruction Following**: Excels at structured tool use and multi-step reasoning required for ReAct agent
- **Query Classification**: Accurately distinguishes structured/unstructured/out-of-scope queries
- **Cost-Efficiency**: Optimized pricing on Nebius makes it practical for iterative development
- **Reasoning Depth**: 235B parameter model provides sufficient capability without over-provisioning
- **All LLM calls** use Nebius Token Factory models as per requirements

**Single Model Approach**: We use one model for all roles (classification + reasoning) to keep the pipeline simple while maintaining performance across all query types.

### Persistence

- SQLite checkpoints in `checkpoints.db`
- Full conversation history per session
- Restored on restart with same session ID

## Tools (7 Available)

| Tool | Purpose | Parameters |
|------|---------|-----------|
| `get_examples` | Fetch dataset rows | n, offset, category, intent |
| `get_distribution` | Value counts per column | column |
| `count_rows` | Count filtered rows | category, intent |
| `search_keyword` | Substring search | keyword, column |
| `get_categories` | List all categories | - |
| `get_intents` | List intents by category | category |
| `get_stats` | Dataset statistics | - |

## Dataset

26,872 rows across 11 categories and 27 intents:

| Category | Intents |
|----------|---------|
| ACCOUNT | create_account, delete_account, edit_account, recover_password, registration_problems, switch_account |
| CANCEL | check_cancellation_fee |
| CONTACT | contact_customer_service, contact_human_agent |
| DELIVERY | delivery_options, delivery_period |
| FEEDBACK | complaint, review |
| INVOICE | check_invoice, get_invoice |
| ORDER | cancel_order, change_order, place_order, track_order |
| PAYMENT | check_payment_methods, payment_issue |
| REFUND | check_refund_policy, get_refund, track_refund |
| SHIPPING | change_shipping_address, set_up_shipping_address |
| SUBSCRIPTION | newsletter_subscription |

## MCP Server & Client

FastMCP server exposes 7 tools for AI clients via Model Context Protocol.

**Start server:**
```bash
python mcp_server.py
```

**Test with client (in another terminal):**
```bash
python mcp_client.py
```

**Available tools:**
- `get_examples(n, offset, category, intent)` - Fetch dataset rows
- `get_distribution(column)` - Value counts for column
- `count_rows(category, intent)` - Count filtered rows
- `search_keyword(keyword, column)` - Substring search
- `get_categories()` - List all categories
- `get_intents(category)` - List intents
- `get_stats()` - Dataset statistics

## Project Structure

```
agent.py              # ReAct agent (LangGraph)
main.py               # CLI entry point
streamlit_app.py      # Web UI (Streamlit)  
mcp_server.py         # MCP server (FastMCP)
mcp_client.py         # MCP client (for testing)
data_analysis_tools.py # Dataset analyzer
requirements.txt      # Dependencies
README.md             # This file
```
