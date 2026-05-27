"""REST API server exposing dataset analysis tools."""

from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_analysis_tools import DatasetAnalyzer

load_dotenv()

app = FastAPI(
    title="Data Analyst API",
    description="API for analyzing the Bitext customer support dataset",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = DatasetAnalyzer()


# Request schemas
class GetExamplesRequest(BaseModel):
    n: int = 5
    offset: int = 0
    category: Optional[str] = None
    intent: Optional[str] = None


class GetDistributionRequest(BaseModel):
    column: str


class CountRowsRequest(BaseModel):
    category: Optional[str] = None
    intent: Optional[str] = None


class SearchKeywordRequest(BaseModel):
    keyword: str
    column: str = "instruction"


class GetIntentsRequest(BaseModel):
    category: Optional[str] = None


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "data-analyst-api"}


@app.get("/tools")
def list_tools():
    """List all available tools."""
    return {
        "tools": [
            {"name": "get_examples", "description": "Fetch dataset rows"},
            {"name": "get_distribution", "description": "Value counts per column"},
            {"name": "count_rows", "description": "Count filtered rows"},
            {"name": "search_keyword", "description": "Substring search"},
            {"name": "get_categories", "description": "List all categories"},
            {"name": "get_intents", "description": "List intents by category"},
            {"name": "get_stats", "description": "Dataset statistics"},
        ]
    }


@app.post("/tools/get_examples")
def get_examples(req: GetExamplesRequest):
    """Return up to n examples from the dataset."""
    filters = {k: v for k, v in {"category": req.category, "intent": req.intent}.items() if v}
    return {"results": analyzer.get_examples(req.n, offset=req.offset, **filters)}


@app.post("/tools/get_distribution")
def get_distribution(req: GetDistributionRequest):
    """Return value counts for a column."""
    return {"distribution": analyzer.get_distribution(req.column)}


@app.post("/tools/count_rows")
def count_rows(req: CountRowsRequest):
    """Count rows in the dataset."""
    filters = {k: v for k, v in {"category": req.category, "intent": req.intent}.items() if v}
    return {"count": analyzer.count(**filters)}


@app.post("/tools/search_keyword")
def search_keyword(req: SearchKeywordRequest):
    """Search for rows where a column contains a keyword."""
    rows = analyzer.search(req.keyword, req.column)
    total = len(rows)
    return {
        "results": rows[:10],
        "total": total,
        "extra": f"[{total - 10} more entries]" if total > 10 else None
    }


@app.post("/tools/get_categories")
def get_categories():
    """Get all unique categories in the dataset."""
    return {"categories": analyzer.get_categories()}


@app.post("/tools/get_intents")
def get_intents(req: GetIntentsRequest):
    """Get all intents, optionally filtered by category."""
    return {"intents": analyzer.get_intents(req.category)}


@app.post("/tools/get_stats")
def get_stats():
    """Get dataset statistics."""
    return {"stats": analyzer.get_stats()}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Data Analyst API on http://0.0.0.0:8000")
    print("📚 Available endpoints:")
    print("   POST /tools/get_examples")
    print("   POST /tools/get_distribution")
    print("   POST /tools/count_rows")
    print("   POST /tools/search_keyword")
    print("   POST /tools/get_categories")
    print("   POST /tools/get_intents")
    print("   POST /tools/get_stats")
    print("🏥 Health check: GET /health")
    print("📖 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
