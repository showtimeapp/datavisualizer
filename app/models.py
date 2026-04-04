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
    temporary: bool = False
    preview: list[dict[str, Any]] = []


class UploadResponse(BaseModel):
    dataset_ids: list[str]              # Quick access to IDs
    datasets: list[DatasetInfo]         # Full details
    message: str


# ─── Text Input (temporary session) ──────────────────────

class TextInputRequest(BaseModel):
    """Send raw text — system extracts data, stores temporarily (auto-deletes)."""
    text: str = Field(
        ...,
        description="Raw text containing data — can be CSV-like, sentences with numbers, or any format",
        examples=[
            "Revenue was $4.2M in Q1, $3.8M in Q2, and $5.1M in Q3",
            "AAPL,150.2,155.3,149.0,154.8\nGOOG,2800,2850,2790,2840",
            "Company A profit 12%, Company B profit 8%, Company C profit 15%",
        ],
    )
    name: str = Field(default="text_input", description="Optional label for this data")


class TextInputResponse(BaseModel):
    dataset_id: str
    temporary: bool = True
    expires_in_seconds: int
    columns: list[ColumnInfo]
    row_count: int
    preview: list[dict[str, Any]] = []
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
    chart = "chart"
    analysis = "analysis"
    query = "query"
    clarify = "clarify"


class ChatResponse(BaseModel):
    """Unified response — can contain a chart, analysis, answer, or a clarification question."""
    intent: IntentType
    message: str

    # Chart output (only when intent = "chart")
    chart: Optional[dict[str, Any]] = None
    chart_type: Optional[str] = None
    chart_config: Optional[dict] = None

    # Analysis output (only when intent = "analysis")
    analysis: Optional[dict[str, Any]] = None

    # Query output (only when intent = "query")
    data: Optional[Any] = None

    # Clarification (only when intent = "clarify")
    options: Optional[list[dict]] = None

    # Always present
    datasets_used: list[str] = []