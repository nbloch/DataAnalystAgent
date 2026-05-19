import os
from enum import Enum
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, ConfigDict, Field
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
# Tool input schemas
# ---------------------------------------------------------------------------

class GetCategoriesInput(BaseModel):
    pass

class GetIntentsInput(BaseModel):
    category: str | None = Field(default=None, description="Filter by category (case-sensitive). If None, returns all intents.")

class GetDistributionInput(BaseModel):
    column: str = Field(description="Column to compute distribution for. Common values: 'category', 'intent', 'flags'.")

class GetExamplesInput(BaseModel):
    n: int = Field(default=5, description="Maximum number of examples to return.")
    category: str | None = Field(default=None, description="Filter by category.")
    intent: str | None = Field(default=None, description="Filter by intent.")

class CountRowsInput(BaseModel):
    category: str | None = Field(default=None, description="Filter by category.")
    intent: str | None = Field(default=None, description="Filter by intent.")

class SearchKeywordInput(BaseModel):
    keyword: str = Field(description="Substring to search for (case-insensitive).")
    column: str = Field(default="instruction", description="Column to search in.")

class GetStatsInput(BaseModel):
    pass

# ---------------------------------------------------------------------------
# Tool output types
# ---------------------------------------------------------------------------

class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    instruction: str
    response: str
    category: str
    intent: str

class DatasetStats(BaseModel):
    num_rows: int
    columns: list[str]
    unique_counts: dict[str, int]
    avg_instruction_length: float
    avg_response_length: float

# ---------------------------------------------------------------------------
# Dataset tools
# ---------------------------------------------------------------------------

_analyzer = DatasetAnalyzer()

def _get_categories() -> list[str]:
    return _analyzer.get_categories()

def _get_intents(category: str | None = None) -> list[str]:
    return _analyzer.get_intents(category)

def _get_distribution(column: str) -> dict[str, int]:
    return _analyzer.get_distribution(column)

def _get_examples(n: int = 5, category: str | None = None, intent: str | None = None) -> list[DatasetRow]:
    filters = {k: v for k, v in {"category": category, "intent": intent}.items() if v is not None}
    return [DatasetRow(**row) for row in _analyzer.get_examples(n, **filters)]

def _count_rows(category: str | None = None, intent: str | None = None) -> int:
    filters = {k: v for k, v in {"category": category, "intent": intent}.items() if v is not None}
    return _analyzer.count(**filters)

def _search_keyword(keyword: str, column: str = "instruction") -> list[DatasetRow]:
    return [DatasetRow(**row) for row in _analyzer.search(keyword, column)]

def _get_stats() -> DatasetStats:
    return DatasetStats(**_analyzer.get_stats())

get_categories   = StructuredTool(name="get_categories",   description="Return all unique categories in the dataset.",                                              func=_get_categories,   args_schema=GetCategoriesInput)
get_intents      = StructuredTool(name="get_intents",      description="Return unique intents, optionally filtered by category.",                                   func=_get_intents,      args_schema=GetIntentsInput)
get_distribution = StructuredTool(name="get_distribution", description="Return value counts for a column. Common columns: 'category', 'intent', 'flags'.",          func=_get_distribution, args_schema=GetDistributionInput)
get_examples     = StructuredTool(name="get_examples",     description="Return up to n examples, optionally filtered by category and/or intent.",                   func=_get_examples,     args_schema=GetExamplesInput)
count_rows       = StructuredTool(name="count_rows",       description="Count rows in the dataset, optionally filtered by category and/or intent.",                 func=_count_rows,       args_schema=CountRowsInput)
search_keyword   = StructuredTool(name="search_keyword",   description="Search for rows where a column contains a keyword (case-insensitive).",                     func=_search_keyword,   args_schema=SearchKeywordInput)
get_stats        = StructuredTool(name="get_stats",        description="Return dataset statistics: row count, columns, unique counts, avg message lengths.",         func=_get_stats,        args_schema=GetStatsInput)

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
- Structured: questions with a specific, enumerable answer retrievable directly from the dataset — counts, distributions, lists of categories/intents, specific example lookups (e.g. "How many rows?", "What intents exist?", "Show me 3 examples")
- Unstructured: qualitative or interpretive questions about the dataset content that require reading and analyzing the text — patterns, themes, tone, writing style, sentiment, quality, comparisons, open-ended summaries (e.g. "What themes emerge?", "Describe the tone", "How do agents handle X?", "Summarize the FEEDBACK category")
- Out-of-scope: questions entirely unrelated to the Bitext customer support dataset

Note: questions about how agents behave, write, or handle situations are IN SCOPE if they refer to what the dataset contains.

User request: {user_input}"""),
        ])
        chain = prompt | self.llm.with_structured_output(ClassifiedRequest)
        return chain.invoke({"user_input": user_input}).category

    def _build_graph(self):
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

        def unstructured_node(state: AgentState) -> dict:
            return react_func(state, 20)

        def react_func(state: AgentState, recursion_limit) -> dict:
            user_msg = next(m for m in state["messages"] if isinstance(m, HumanMessage)).content
            last_ai_content = None
            try:
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
            except GraphRecursionError:
                return {"messages": [AIMessage(content="I'm sorry, I wasn't able to complete the answer within the allowed number of steps.")]}
            return {"messages": [AIMessage(content=last_ai_content or "I'm sorry, I couldn't find an answer.")]}


        def route_classify(state: AgentState) -> str:
            return {
                RequestCategory.OUT_OF_SCOPE: "reject",
                RequestCategory.STRUCTURED: "structured",
                RequestCategory.UNSTRUCTURED: "unstructured",
            }[state["category"]]

        builder = StateGraph(AgentState)
        builder.add_node("classify", classify_node)
        builder.add_node("reject", reject_node)
        builder.add_node("structured", structured_node)
        builder.add_node("unstructured", unstructured_node)

        builder.add_edge(START, "classify")
        builder.add_conditional_edges("classify", route_classify, ["reject", "structured", "unstructured"])
        builder.add_edge("reject", END)
        builder.add_edge("structured", END)
        builder.add_edge("unstructured", END)

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
                    if node_name in ("structured", "reject", "react"):
                        for msg in msgs:
                            if msg.content:
                                reply = msg.content
            print(f"Assistant: {reply or 'I could not find an answer.'}\n")


def make_graph(config=None):
    return DataAnalystAgent().graph

if __name__ == "__main__":
    DataAnalystAgent().run()
