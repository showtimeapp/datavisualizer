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
    role: str
    sample_values: list[str] = []


class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    columns: list[ColumnInfo]
    row_count: int
    temporary: bool = False
    preview: list[dict[str, Any]] = []


class UploadResponse(BaseModel):
    dataset_ids: list[str]
    datasets: list[DatasetInfo]
    message: str


# ─── Text Input (all-in-one: data + question → answer) ──

class TextRequest(BaseModel):
    """Send raw text data + a question. System parses data, answers question, returns result — all in one call."""
    text: str = Field(
        ...,
        description="Raw text containing data",
        examples=[
            "AAPL open 150 close 155, GOOG open 2800 close 2850",
            "Company,Revenue,Profit\nApple,394B,99B\nGoogle,307B,73B",
            "Revenue was $4.2M in Q1, $3.8M in Q2, and $5.1M in Q3",
        ],
    )
    question: str = Field(
        ...,
        description="What you want to know or see from the data",
        examples=[
            "which company has highest profit?",
            "bar chart comparing revenue",
            "summary statistics",
            "list all columns",
        ],
    )
    name: str = Field(default="text_input", description="Optional label for this data")


class TextResponse(BaseModel):
    """Response from the all-in-one text endpoint."""
    intent: str
    message: str
    chart_png: Optional[str] = None       # Base64 PNG image
    chart: Optional[dict[str, Any]] = None # Plotly JSON (fallback)
    chart_type: Optional[str] = None
    data: Optional[Any] = None
    parsed_data: dict = {}                 # Info about the parsed input


# ─── Unified Chat ────────────────────────────────────────

class ChatRequest(BaseModel):
    """The ONE endpoint. User sends a message, system figures out everything."""
    message: str = Field(
        ...,
        description="Natural language message — ask anything about your data",
    )
    dataset_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional: specify which dataset(s) to use.",
    )


class IntentType(str, Enum):
    chart = "chart"
    analysis = "analysis"
    query = "query"
    clarify = "clarify"


class ChatResponse(BaseModel):
    """Unified response — can contain a chart (PNG), analysis, answer, or clarification."""
    intent: IntentType
    message: str

    # Chart output
    chart_png: Optional[str] = None       # Base64 PNG — use this to display
    chart: Optional[dict[str, Any]] = None # Plotly JSON — fallback / interactive use
    chart_type: Optional[str] = None
    chart_config: Optional[dict] = None

    # Analysis output
    analysis: Optional[dict[str, Any]] = None

    # Query output
    data: Optional[Any] = None

    # Clarification
    options: Optional[list[dict]] = None

    # Always present
    datasets_used: list[str] = []