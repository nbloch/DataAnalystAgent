# Nebius Data Analyst Agent

A conversational LangGraph agent for exploring the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), powered by DeepSeek-V3.2 via Nebius AI.

## Architecture

The agent uses a LangGraph state machine with three paths:

```
User input
    │
    ▼
[classify]  ──▶  Out-of-scope  ──▶  [reject]
    │
    ├──▶  Structured    ──▶  [ReAct agent, limit 10 steps]
    │
    └──▶  Unstructured  ──▶  [ReAct agent, limit 20 steps]
```

- **Structured**: questions with enumerable answers (counts, distributions, example lookups) — answered with dataset tools
- **Unstructured**: qualitative/interpretive questions (themes, tone, writing style) — answered with deeper ReAct reasoning
- **Out-of-scope**: questions unrelated to the dataset — politely rejected

Conversation history is persisted across sessions via SQLite (`checkpoints.db`).

## Dataset

The Bitext customer support dataset contains customer–agent conversation pairs across 11 categories and 27 intents:

| Category     | Intents |
|-------------|---------|
| ACCOUNT     | create_account, delete_account, edit_account, recover_password, registration_problems, switch_account |
| CANCEL      | check_cancellation_fee |
| CONTACT     | contact_customer_service, contact_human_agent |
| DELIVERY    | delivery_options, delivery_period |
| FEEDBACK    | complaint, review |
| INVOICE     | check_invoice, get_invoice |
| ORDER       | cancel_order, change_order, place_order, track_order |
| PAYMENT     | check_payment_methods, payment_issue |
| REFUND      | check_refund_policy, get_refund, track_refund |
| SHIPPING    | change_shipping_address, set_up_shipping_address |
| SUBSCRIPTION| newsletter_subscription |

## Tools

| Tool | Description |
|------|-------------|
| `get_distribution` | Value counts for a column (`category`, `intent`, `flags`) |
| `get_examples` | Return up to n rows, with optional category/intent filter and pagination |
| `count_rows` | Count rows, optionally filtered |
| `search_keyword` | Case-insensitive substring search on any column (returns up to 10 results) |
| `get_stats` | Row count, column names, unique counts, avg message lengths |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
NEBIUS_API_KEY=your_key_here
```

## Usage

```bash
# Single session (no memory persistence)
python main.py

# Named session (conversation persists in checkpoints.db)
python main.py --session my_session
```

### Example queries

```
How many rows are in the dataset?
What are all the intents in the ORDER category?
Show me 3 examples of complaints.
How do agents typically handle refund requests?
What's the distribution of categories?
```
