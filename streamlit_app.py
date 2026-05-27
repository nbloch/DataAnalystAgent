"""Streamlit web interface for the Data Analyst Agent."""

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from agent import DataAnalystAgent

st.set_page_config(
    page_title="Data Analyst Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Bitext Data Analyst Agent")
st.markdown("Explore customer support data through natural conversation.")

# Initialize agent once per session
if "agent" not in st.session_state:
    st.session_state.agent = DataAnalystAgent()

agent = st.session_state.agent

# Sidebar: Session management
with st.sidebar:
    st.header("⚙️ Session Management")

    session_id = st.text_input(
        "Session ID",
        value="default",
        help="Use the same ID to restore previous conversations"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Load", use_container_width=True):
            st.session_state.current_session = session_id
            st.session_state.messages = []
            st.success(f"Loaded session: {session_id}")

    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.success("History cleared!")

    st.divider()
    st.header("💡 Example Queries")

    examples = [
        "What categories exist in the dataset?",
        "How many refund requests did we get?",
        "Show me 4 examples of the REFUND category",
        "Show me 2 more",
        "Summarize how agents respond to complaint intents",
        "What is the distribution of intents in the ACCOUNT category?",
        "Show me examples of people wanting their money back"
    ]

    for i, example in enumerate(examples):
        if st.button(example, use_container_width=True, key=f"example_{i}"):
            st.session_state.input_query = example
            st.rerun()

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_session" not in st.session_state:
    st.session_state.current_session = "default"

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask about the dataset...")

if user_input or "input_query" in st.session_state:
    # Get either fresh input or preset example
    current_query = user_input if user_input else st.session_state.pop("input_query", None)

    if current_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(current_query)

        st.session_state.messages.append({"role": "user", "content": current_query})

        # Get agent response
        with st.spinner("🤔 Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.current_session}}

            assistant_response = None
            reasoning_steps = []

            try:
                for chunk in agent.graph.stream(
                    {"messages": [HumanMessage(content=current_query)], "category": None},
                    config,
                    stream_mode="updates",
                ):
                    for node_name, update in chunk.items():
                        msgs = update.get("messages", [])
                        if node_name in ("structured", "reject", "unstructured"):
                            for msg in msgs:
                                if msg.content:
                                    assistant_response = msg.content
            except Exception as e:
                assistant_response = f"Error: {str(e)}"

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_response or "I could not find an answer.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response or "I could not find an answer."
        })

        st.rerun()

# Sidebar: Conversation info
with st.sidebar:
    st.divider()
    st.subheader("📋 Conversation Info")
    st.write(f"**Session**: {st.session_state.current_session}")
    st.write(f"**Messages**: {len(st.session_state.messages)}")
    st.write(f"**Updated**: {datetime.now().strftime('%H:%M:%S')}")
