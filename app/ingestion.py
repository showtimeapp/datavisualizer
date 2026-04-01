"""
Ingestion — accepts CSV, Excel, HTML, Markdown, JSON, and raw text.
Converts everything into a pandas DataFrame.
"""

import io
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from app.llm_client import call_llm, extract_json_from_response


# ─── Dispatcher ──────────────────────────────────────────

async def ingest_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Detect file type and parse into a DataFrame."""
    ext = Path(filename).suffix.lower()

    parsers = {
        ".csv": parse_csv,
        ".tsv": parse_tsv,
        ".xlsx": parse_excel,
        ".xls": parse_excel,
        ".html": parse_html,
        ".htm": parse_html,
        ".md": parse_markdown,
        ".json": parse_json,
        ".txt": parse_text,
    }

    parser = parsers.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext}")

    # Text parser needs LLM (async), others are sync
    if ext == ".txt":
        return await parse_text(file_bytes)

    return parser(file_bytes)


async def ingest_raw_text(text: str) -> pd.DataFrame:
    """Parse unstructured text into a DataFrame using LLM."""
    return await parse_text(text.encode("utf-8"))


# ─── Parsers ─────────────────────────────────────────────

def parse_csv(data: bytes) -> pd.DataFrame:
    # Try common encodings
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV with common encodings")


def parse_tsv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), sep="\t")


def parse_excel(data: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    # If multiple sheets, read first one; return dict info in metadata
    df = pd.read_excel(xls, sheet_name=0)
    return df


def parse_html(data: bytes) -> pd.DataFrame:
    """Extract the largest table from an HTML file."""
    soup = BeautifulSoup(data, "lxml")
    tables = pd.read_html(io.BytesIO(data))
    if not tables:
        raise ValueError("No tables found in HTML file")
    # Return the largest table
    return max(tables, key=len)


def parse_markdown(data: bytes) -> pd.DataFrame:
    """Parse a markdown table into a DataFrame."""
    text = data.decode("utf-8")

    # Find markdown table lines (lines with pipes)
    lines = text.strip().split("\n")
    table_lines = [l for l in lines if "|" in l]

    if len(table_lines) < 2:
        raise ValueError("No markdown table found in file")

    # Parse header
    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.strip("|").split("|")]

    # Skip separator line (---|---|---)
    data_lines = []
    for line in table_lines[1:]:
        stripped = line.strip("| ").replace("-", "").replace("|", "").strip()
        if not stripped:
            continue  # separator line
        cells = [c.strip() for c in line.strip("|").split("|")]
        data_lines.append(cells)

    if not data_lines:
        raise ValueError("Markdown table has no data rows")

    df = pd.DataFrame(data_lines, columns=headers)
    return df


def parse_json(data: bytes) -> pd.DataFrame:
    """Parse JSON into DataFrame — handles arrays and nested objects."""
    import json as json_mod
    parsed = json_mod.loads(data)

    if isinstance(parsed, list):
        return pd.json_normalize(parsed)
    elif isinstance(parsed, dict):
        # Check for common wrapper patterns: {"data": [...]} etc.
        for key in ["data", "results", "records", "rows", "items"]:
            if key in parsed and isinstance(parsed[key], list):
                return pd.json_normalize(parsed[key])
        # Flat dict → single-row DataFrame
        return pd.json_normalize(parsed)
    else:
        raise ValueError("JSON root must be an array or object")


async def parse_text(data: bytes) -> pd.DataFrame:
    """Use LLM to extract structured data from unstructured text."""
    text = data.decode("utf-8")

    system_prompt = """You are a financial data extraction engine.
Given unstructured text, extract ALL numerical/financial data into a JSON table.

Rules:
- Return ONLY valid JSON, no markdown, no explanation.
- Format: {"columns": ["Col1", "Col2", ...], "data": [[val1, val2, ...], ...]}
- Use clear column names (e.g., "Quarter", "Revenue_USD", "Profit_Margin_Pct")
- Convert percentages to numbers (e.g., 15% → 15)
- Remove currency symbols, keep the number (e.g., $4.2M → 4200000)
- If the text has no extractable data, return: {"columns": [], "data": []}
"""

    response = await call_llm(system_prompt, text)
    parsed = extract_json_from_response(response)

    if not parsed.get("columns") or not parsed.get("data"):
        raise ValueError("LLM could not extract structured data from the text")

    df = pd.DataFrame(parsed["data"], columns=parsed["columns"])
    return df
