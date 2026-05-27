# Bitext Data Analyst Agent

A conversational AI agent for analyzing the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Available via CLI, web UI, and REST API.

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

**API** - REST endpoints for integrations
```bash
python mcp_server.py
# Open http://localhost:8000/docs for interactive API docs
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

### Model

- **Qwen/Qwen3-235B-A22B-Instruct-2507** via Nebius Token Factory
  - Strong instruction following
  - Fast response times
  - Cost-efficient

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

## API Reference

All tools available as REST endpoints via FastAPI. Visit `http://localhost:8000/docs` for interactive documentation.

```bash
curl -X POST http://localhost:8000/tools/get_examples \
  -H "Content-Type: application/json" \
  -d '{"n": 5, "category": "REFUND"}'
```

**Endpoints:** `get_examples` | `get_distribution` | `count_rows` | `search_keyword` | `get_categories` | `get_intents` | `get_stats` | `health`

## Project Structure

```
agent.py              # ReAct agent (LangGraph)
main.py               # CLI entry point
streamlit_app.py      # Web UI (Streamlit)  
mcp_server.py         # REST API (FastAPI)
data_analysis_tools.py # Dataset analyzer
requirements.txt      # Dependencies
```
