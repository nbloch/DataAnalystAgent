import os
from enum import Enum
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import BaseModel
from typing_extensions import TypedDict

from data_analysis_tools import DatasetAnalyzer


# ---------------------------------------------------------------------------
# Classification types
# ---------------------------------------------------------------------------

class RequestCategory(Enum):
    STRUCTURED = "Structured"
    UNSTRUCTURED = "Unstructured"
    OUT_OF_SCOPE = "Out-of-scope"


class ClassifiedRequest(BaseModel):
    category: RequestCategory


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    category: RequestCategory | None


# ---------------------------------------------------------------------------
# Dataset tools
# ---------------------------------------------------------------------------

_analyzer = DatasetAnalyzer()


@tool
def get_categories() -> list[str]:
    """Return all unique categories in the dataset."""
    return _analyzer.get_categories()


@tool
def get_intents(category: str | None = None) -> list[str]:
    """Return unique intents, optionally filtered by category."""
    return _analyzer.get_intents(category)


@tool
def get_distribution(column: str) -> dict:
    """Return value counts for a column. Common columns: 'category', 'intent', 'flags'."""
    return _analyzer.get_distribution(column)


@tool
def get_examples(n: int = 5, category: str | None = None, intent: str | None = None) -> list[dict]:
    """Return up to n examples, optionally filtered by category and/or intent."""
    filters = {k: v for k, v in {"category": category, "intent": intent}.items() if v is not None}
    return _analyzer.get_examples(n, **filters)


@tool
def count_rows(category: str | None = None, intent: str | None = None) -> int:
    """Count rows in the dataset, optionally filtered by category and/or intent."""
    filters = {k: v for k, v in {"category": category, "intent": intent}.items() if v is not None}
    return _analyzer.count(**filters)


@tool
def search_keyword(keyword: str, column: str = "instruction") -> list[dict]:
    """Search for rows where a column contains a keyword (case-insensitive)."""
    return _analyzer.search(keyword, column)


@tool
def get_stats() -> dict:
    """Return dataset statistics: row count, columns, unique counts, avg message lengths."""
    return _analyzer.get_stats()


TOOLS = [get_categories, get_intents, get_distribution, get_examples, count_rows, search_keyword, get_stats]

# ---------------------------------------------------------------------------
# ReAct prompt (explicit Thought / Action / Observation format)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DataAnalystAgent:
    MODEL = "deepseek-ai/DeepSeek-V3.2"
    SYSTEM_PROMPT = """
You are a data analyst agent. Answer the user requests about the Bitext customer support dataset.
"""

    def __init__(self):
        self.llm = ChatOpenAI(
            base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
            model=self.MODEL,
        )
        self.graph = self._build_graph()

    def _classify(self, user_input: str) -> RequestCategory:
        # FIXME: Improve performance on that
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("user", """Classify the following user request into exactly one of these categories:
- Structured: concrete, data-driven questions about the dataset (counts, distributions, examples, categories)
- Unstructured: open-ended questions requiring summarization or interpretation of the dataset
- Out-of-scope: questions unrelated to the dataset

User request: {user_input}"""),
        ])
        chain = prompt | self.llm.with_structured_output(ClassifiedRequest)
        return chain.invoke({"user_input": user_input}).category

    def _build_graph(self):
        llm_forced_tool = self.llm.bind_tools(TOOLS, tool_choice="required")

        react_subgraph = create_react_agent(self.llm, TOOLS, prompt=self.SYSTEM_PROMPT)

        def classify_node(state: AgentState) -> dict:
            category = self._classify(state["messages"][-1].content)
            return {"category": category}

        def reject_node(state: AgentState) -> dict:
            return {"messages": [AIMessage(
                content="I'm sorry, that question is outside the scope of this dataset. "
                        "I can only help with questions about the Bitext customer support dataset."
            )]}

        def structured_node(state: AgentState) -> dict:
            return react_func(state, 10)


        def structured_answer_node(state: AgentState) -> dict:
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + state["messages"]
            return {"messages": [self.llm.bind_tools(TOOLS, tool_choice="none").invoke(messages)]}
        
        def unstructured_node(state: AgentState) -> dict:
            return react_func(state, 20)
        
        def react_func(state: AgentState, recursion_limit) -> dict:
            user_msg = next(m for m in state["messages"] if isinstance(m, HumanMessage)).content
            last_ai_content = None
            for chunk in react_subgraph.stream(
                {"messages": [HumanMessage(content=user_msg)]},
                {"recursion_limit": recursion_limit},
                stream_mode="updates",
            ):
                for node_name, update in chunk.items():
                    msgs = update.get("messages", [])
                    if node_name == "agent":
                        for msg in msgs:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                                    print(f"  [tool call] {tc['name']}({args_str})")
                            elif isinstance(msg, AIMessage) and msg.content:
                                last_ai_content = msg.content
                    elif node_name == "tools":
                        for msg in msgs:
                            print(f"  [tool result] {str(msg.content)[:300]}")
            return {"messages": [AIMessage(content=last_ai_content or "I'm sorry, I couldn't find an answer.")]}


        def route_classify(state: AgentState) -> str:
            return {
                RequestCategory.OUT_OF_SCOPE: "reject",
                RequestCategory.STRUCTURED: "structured",
                RequestCategory.UNSTRUCTURED: "react",
            }[state["category"]]

        builder = StateGraph(AgentState)
        builder.add_node("classify", classify_node)
        builder.add_node("reject", reject_node)
        builder.add_node("structured", structured_node)
        builder.add_node("structured_tool", ToolNode(TOOLS))
        builder.add_node("structured_answer", structured_answer_node)
        builder.add_node("react", unstructured_node)

        builder.add_edge(START, "classify")
        builder.add_conditional_edges("classify", route_classify, ["reject", "structured", "react"])
        builder.add_edge("reject", END)
        builder.add_edge("structured", "structured_tool")
        builder.add_edge("structured_tool", "structured_answer")
        builder.add_edge("structured_answer", END)
        builder.add_edge("react", END)

        return builder.compile()

    def run(self):
        print("Agent ready. Type 'quit' to exit.\n")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input:
                continue

            reply = None
            for chunk in self.graph.stream(
                {"messages": [HumanMessage(content=user_input)], "category": None},
                stream_mode="updates",
            ):
                for node_name, update in chunk.items():
                    msgs = update.get("messages", [])
                    if node_name == "classify":
                        print(f"  [classify] {update['category'].value}")
                    elif node_name == "structured":
                        for msg in msgs:
                            if hasattr(msg, "tool_calls"):
                                for tc in msg.tool_calls:
                                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                                    print(f"  [tool call] {tc['name']}({args_str})")
                    elif node_name == "structured_tool":
                        for msg in msgs:
                            print(f"  [tool result] {str(msg.content)[:300]}")
                    elif node_name in ("structured_answer", "reject", "react"):
                        for msg in msgs:
                            if msg.content:
                                reply = msg.content
            print(f"Assistant: {reply or 'I could not find an answer.'}\n")


def make_graph(config=None):
    return DataAnalystAgent().graph

if __name__ == "__main__":
    DataAnalystAgent().run()
