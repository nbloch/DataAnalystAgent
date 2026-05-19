import os
from enum import Enum
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
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
    tool_calls_count: int


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
        llm_with_tools = self.llm.bind_tools(TOOLS)
        llm_forced_tool = self.llm.bind_tools(TOOLS, tool_choice="required")

        def classify_node(state: AgentState) -> dict:
            category = self._classify(state["messages"][-1].content)
            return {"category": category, "tool_calls_count": 0}

        def reject_node(state: AgentState) -> dict:
            return {"messages": [AIMessage(
                content="I'm sorry, that question is outside the scope of this dataset. "
                        "I can only help with questions about the Bitext customer support dataset."
            )]}

        def structured_node(state: AgentState) -> dict:
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + state["messages"]
            return {"messages": [llm_forced_tool.invoke(messages)]}

        def structured_answer_node(state: AgentState) -> dict:
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + state["messages"]
            return {"messages": [self.llm.invoke(messages)]}

        def react_node(state: AgentState) -> dict:
            messages = [SystemMessage(content=self.SYSTEM_PROMPT)] + state["messages"]
            tool_calls_count = state.get("tool_calls_count", 0)
            # Once the tool call limit is reached, force a text answer
            llm = self.llm if tool_calls_count >= 3 else llm_with_tools
            response = llm.invoke(messages)
            new_count = tool_calls_count + (1 if getattr(response, "tool_calls", None) else 0)
            return {"messages": [response], "tool_calls_count": new_count}

        def route_classify(state: AgentState) -> str:
            return {
                RequestCategory.OUT_OF_SCOPE: "reject",
                RequestCategory.STRUCTURED: "structured",
                RequestCategory.UNSTRUCTURED: "react",
            }[state["category"]]

        def route_react(state: AgentState) -> str:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "react_tool"
            return END

        builder = StateGraph(AgentState)
        builder.add_node("classify", classify_node)
        builder.add_node("reject", reject_node)
        builder.add_node("structured", structured_node)
        builder.add_node("structured_tool", ToolNode(TOOLS))
        builder.add_node("structured_answer", structured_answer_node)
        builder.add_node("react", react_node)
        builder.add_node("react_tool", ToolNode(TOOLS))

        builder.add_edge(START, "classify")
        builder.add_conditional_edges("classify", route_classify, ["reject", "structured", "react"])
        builder.add_edge("reject", END)
        builder.add_edge("structured", "structured_tool")
        builder.add_edge("structured_tool", "structured_answer")
        builder.add_edge("structured_answer", END)
        builder.add_conditional_edges("react", route_react, ["react_tool", END])
        builder.add_edge("react_tool", "react")

        return builder.compile()

    def run(self):
        print("Agent ready. Type 'quit' to exit.\n")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input:
                continue

            result = self.graph.invoke({
                "messages": [HumanMessage(content=user_input)],
                "category": None,
                "tool_calls_count": 0,
            })

            last_ai = next(
                (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
                None,
            )
            reply = last_ai.content if last_ai else "I'm sorry, I couldn't find an answer to your question."
            print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    DataAnalystAgent().run()
