"""
Pydantic models — unified chat-based API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum


# ─── Dataset ─────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    role: str  # "dimension" or "measure"
    sample_values: list[str] = []


class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    columns: list[ColumnInfo]
    row_count: int
    preview: list[dict[str, Any]] = []


class UploadResponse(BaseModel):
    datasets: list[DatasetInfo]
    message: str


# ─── Unified Chat ────────────────────────────────────────

class ChatRequest(BaseModel):
    """The ONE endpoint. User sends a message, system figures out everything."""
    message: str = Field(
        ...,
        description="Natural language message — ask anything about your data",
        examples=[
            "show me a bar chart of revenue by quarter",
            "what was the highest profit month?",
            "give me summary statistics",
            "compare revenue from sales.csv and marketing.csv",
            "scatter plot of price vs volume from the stock data",
        ],
    )
    dataset_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional: specify which dataset(s) to use. If None, system picks or asks.",
    )


class IntentType(str, Enum):
    chart = "chart"           # User wants a visualization
    analysis = "analysis"     # User wants stats/aggregation/comparison
    query = "query"           # User wants to ask a question about the data
    clarify = "clarify"       # System needs to ask the user something


class ChatResponse(BaseModel):
    """Unified response — can contain a chart, analysis, answer, or a clarification question."""
    intent: IntentType
    message: str                             # Human-readable answer/explanation

    # Chart output (only when intent = "chart")
    chart: Optional[dict[str, Any]] = None   # Full Plotly JSON spec
    chart_type: Optional[str] = None
    chart_config: Optional[dict] = None      # What the LLM interpreted

    # Analysis output (only when intent = "analysis")
    analysis: Optional[dict[str, Any]] = None

    # Query output (only when intent = "query")
    data: Optional[Any] = None

    # Clarification (only when intent = "clarify")
    options: Optional[list[dict]] = None     # [{dataset_id, filename, reason}]

    # Always present
    datasets_used: list[str] = []