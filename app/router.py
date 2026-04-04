"""
Brain / Router — ONE LLM call handles everything:
  - Intent classification (chart vs query)
  - Data computation (if derived metrics needed)
  - Chart configuration OR query code generation

Optimized for minimum token usage — no separate intent classifier.
"""

import logging
import math
import traceback
import pandas as pd
import numpy as np

from app.llm_client import call_llm, extract_json_from_response
from app.storage import load_dataset, load_metadata, list_datasets
from app.analysis import natural_language_query, _safe_execute, _serialize_result
from app.charts import build_chart

logger = logging.getLogger(__name__)


def _build_compact_schema(df: pd.DataFrame) -> str:
    """Build a token-efficient schema string."""
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            mn, mx = df[col].min(), df[col].max()
            lines.append(f"  '{col}' ({dtype}) [{mn}..{mx}]")
        else:
            nunique = df[col].nunique()
            samples = df[col].dropna().head(3).tolist()
            lines.append(f"  '{col}' ({dtype}) [{nunique} unique] e.g. {samples}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

async def process_message(message: str, dataset_ids: list[str] | None = None) -> dict:
    """
    One function, one LLM call (for single dataset queries).
    """
    # ── Step 1: Get datasets ──────────────────────────────
    all_datasets = list_datasets()
    if not all_datasets:
        return {
            "intent": "clarify",
            "message": "No datasets uploaded yet. Please upload a file first.",
            "options": None,
            "datasets_used": [],
        }

    # ── Step 2: Resolve dataset (NO LLM for single dataset) ─
    if dataset_ids:
        selected_ids = dataset_ids
    elif len(all_datasets) == 1:
        selected_ids = [all_datasets[0]["dataset_id"]]
    else:
        # Multiple datasets — this is the ONLY case that needs a separate LLM call
        resolved = await _resolve_datasets(message, all_datasets)
        if resolved["needs_clarification"]:
            return {
                "intent": "clarify",
                "message": resolved["message"],
                "options": resolved["options"],
                "datasets_used": [],
            }
        selected_ids = resolved["dataset_ids"]

    # ── Step 3: Load datasets ─────────────────────────────
    datasets = {}
    for did in selected_ids:
        try:
            datasets[did] = {
                "df": load_dataset(did),
                "meta": load_metadata(did),
            }
        except FileNotFoundError:
            return {
                "intent": "clarify",
                "message": f"Dataset '{did}' not found.",
                "options": [{"dataset_id": d["dataset_id"], "filename": d["filename"]}
                           for d in all_datasets],
                "datasets_used": [],
            }

    # ── Step 4: ONE unified LLM call ─────────────────────
    if len(datasets) == 1:
        did = list(datasets.keys())[0]
        df = datasets[did]["df"]
        meta = datasets[did]["meta"]
        return await _unified_handler(message, df, meta["filename"], selected_ids)
    else:
        # Multi-dataset: merge and handle
        df = _merge_datasets(datasets)
        source = ", ".join(ds["meta"]["filename"] for ds in datasets.values())
        return await _unified_handler(message, df, source, selected_ids)


# ═══════════════════════════════════════════════════════════
#  UNIFIED HANDLER — ONE LLM call for intent + execution
# ═══════════════════════════════════════════════════════════

async def _unified_handler(message: str, df: pd.DataFrame, source: str, dataset_ids: list[str]) -> dict:
    """
    Single LLM call that decides intent AND provides the execution plan.
    For charts with computed metrics, it generates the code AND chart config together.
    """
    schema = _build_compact_schema(df)

    system_prompt = """You are an expert financial data analyst and visualization engine.
Given a user message and dataset schema, respond with a SINGLE JSON object.

STEP 1: Decide if the user wants a CHART (visual) or a QUERY (answer/data).
STEP 2: Provide the execution plan.

═══ IF CHART ═══
The user wants to visualize data. They might need computed columns first.

Return:
{
    "intent": "chart",
    "compute_code": "pandas code to prepare data, or null if no computation needed",
    "chart_type": "bar|line|area|pie|donut|scatter|histogram|heatmap|box|waterfall|funnel|radar|treemap",
    "x": "column_name",
    "y": ["col1", "col2"],
    "color": "optional_column_or_null",
    "title": "Descriptive Title",
    "explanation": "What this chart shows"
}

CRITICAL for charts:
- If the user wants a DERIVED metric (e.g., daily return, moving average, ratio), 
  you MUST write compute_code that creates the new column(s) first.
- The x and y columns MUST exist AFTER compute_code runs.
- compute_code uses `df` variable. Create new columns on df directly.
- If no computation needed, set compute_code to null.
- Keep chart data manageable: if result has 1000+ rows, aggregate or limit in compute_code.

Examples:
  "bar chart of revenue by quarter" (no computation needed):
    compute_code: null, x: "Quarter", y: ["Revenue"]
  
  "chart showing daily return (close-open)/open" (needs computation):
    compute_code: "df['DailyReturn'] = ((df['ClsPric'] - df['OpnPric']) / df['OpnPric'] * 100).round(2)\\ndf = df.nlargest(20, 'DailyReturn')",
    x: "TckrSymb", y: ["DailyReturn"]
  
  "line chart of 20-day moving average" (needs computation):
    compute_code: "df = df.sort_values('TradDt')\\ndf['MA20'] = df['ClsPric'].rolling(20).mean().round(2)\\ndf = df.dropna(subset=['MA20'])",
    x: "TradDt", y: ["ClsPric", "MA20"]

═══ IF QUERY ═══
The user wants data, analysis, or an answer.

Return:
{
    "intent": "query",
    "code": "pandas code that assigns result to `result` variable",
    "explanation": "Brief description of what the answer shows"
}

Rules for code:
- Variable `df` is the DataFrame. `pd` and `np` are available.
- MUST assign final answer to `result`.
- `result` can be: string, number, dict, list, DataFrame, or Series.
- For large results, limit to .head(20).
- Round numbers to 2 decimal places.
- NO imports, file I/O, exec, eval, os, sys.

═══ GENERAL RULES ═══
- Use EXACT column names from the schema. Spelling and case must match.
- Return ONLY valid JSON, no markdown, no explanation outside the JSON.
- If user asks to "show" computed data but doesn't mention chart/graph/plot explicitly, prefer "query".
- If user says "show in chart" or "visualize" or "plot", prefer "chart".
- Be concise in compute_code/code — no unnecessary operations."""

    user_prompt = f"""Dataset ({df.shape[0]} rows, {df.shape[1]} cols):
{schema}

Sample (first 3 rows):
{df.head(3).to_string(max_cols=15)}

User: "{message}"
"""

    try:
        response = await call_llm(system_prompt, user_prompt)
        plan = extract_json_from_response(response)
    except Exception as e:
        logger.error(f"Unified handler LLM call failed: {e}")
        # Fallback to query engine
        return await _fallback_to_query(message, df, dataset_ids, str(e))

    intent = plan.get("intent", "query")

    if intent == "chart":
        return await _execute_chart_plan(plan, df, source, dataset_ids)
    else:
        return await _execute_query_plan(plan, df, dataset_ids)


async def _execute_chart_plan(plan: dict, df: pd.DataFrame, source: str, dataset_ids: list[str]) -> dict:
    """Execute a chart plan — compute first if needed, then build chart."""
    try:
        working_df = df.copy()

        # Step 1: Run computation if needed
        compute_code = plan.get("compute_code")
        if compute_code:
            exec_result = _safe_execute(working_df, f"{compute_code}\nresult = df")
            if exec_result["success"]:
                computed = exec_result["result"]
                if isinstance(computed, pd.DataFrame):
                    working_df = computed
                else:
                    logger.warning("compute_code did not return a DataFrame, using original")
            else:
                logger.warning(f"compute_code failed: {exec_result['error']}, trying without computation")

        # Step 2: Build the chart
        chart_type = plan.get("chart_type", "bar")
        x = plan.get("x", "")
        y = plan.get("y", [])
        color = plan.get("color")
        title = plan.get("title", f"{chart_type.title()} Chart")

        # Validate columns exist
        all_cols = set(working_df.columns.tolist())
        if x and x not in all_cols:
            x = _fuzzy_match(x, all_cols)
        y = [(_fuzzy_match(c, all_cols) if c not in all_cols else c) for c in y]

        plotly_result = build_chart(working_df, chart_type, x, y, color, title)

        return {
            "intent": "chart",
            "message": plan.get("explanation", f"Here's your {chart_type} chart."),
            "chart": plotly_result.get("plotly_json"),
            "chart_png": plotly_result.get("png_base64"),
            "chart_type": chart_type,
            "chart_config": {
                "x": x, "y": y, "color": color, "title": title,
                "computed": bool(compute_code),
            },
            "datasets_used": dataset_ids,
        }

    except Exception as e:
        logger.error(f"Chart execution failed: {traceback.format_exc()}")
        # Fall back to query — show data instead of chart
        return {
            "intent": "chart",
            "message": f"Chart generation failed: {str(e)}. Try specifying exact column names from your data.",
            "chart": None,
            "data": {"columns": df.columns.tolist()},
            "datasets_used": dataset_ids,
        }


async def _execute_query_plan(plan: dict, df: pd.DataFrame, dataset_ids: list[str]) -> dict:
    """Execute a query plan — run pandas code and return results."""
    code = plan.get("code", "")
    explanation = plan.get("explanation", "")

    if not code:
        return {
            "intent": "query",
            "message": explanation or "I couldn't determine how to answer this.",
            "data": None,
            "datasets_used": dataset_ids,
        }

    exec_result = _safe_execute(df, code)

    if exec_result["success"]:
        data = _serialize_result(exec_result["result"])
        answer = explanation

        # If result is simple, inline it
        if isinstance(exec_result["result"], (int, float, str, bool)):
            answer = f"{explanation}\n\nAnswer: {exec_result['result']}"

        return {
            "intent": "query",
            "message": answer,
            "data": data,
            "datasets_used": dataset_ids,
        }
    else:
        logger.warning(f"Query code failed: {exec_result['error']}")
        # Try the full fallback in analysis.py
        from app.analysis import _fallback_query
        fallback = await _fallback_query(df, plan.get("explanation", "query"), exec_result["error"])
        return {
            "intent": "query",
            "message": fallback.get("answer", "Query failed."),
            "data": fallback.get("data"),
            "datasets_used": dataset_ids,
        }


async def _fallback_to_query(message: str, df: pd.DataFrame, dataset_ids: list[str], error: str) -> dict:
    """If the unified handler completely fails, fall back to the query engine."""
    try:
        result = await natural_language_query(df, message)
        return {
            "intent": "query",
            "message": result.get("answer", ""),
            "data": result.get("data"),
            "datasets_used": dataset_ids,
        }
    except Exception as e:
        return {
            "intent": "query",
            "message": f"I couldn't process that. Error: {error}. Columns available: {df.columns.tolist()}",
            "data": {"columns": df.columns.tolist()},
            "datasets_used": dataset_ids,
        }


# ─── Dataset Resolution (only for multi-dataset) ─────────

async def _resolve_datasets(message: str, all_datasets: list[dict]) -> dict:
    """Figure out which dataset(s) the user is referring to. Only called when multiple datasets exist."""
    ds_list = "\n".join(
        f"  - id:\"{d['dataset_id']}\" file:\"{d['filename']}\" rows:{d['row_count']}"
        for d in all_datasets
    )

    system_prompt = """Given a user message and available datasets, pick which dataset(s) to use.
Return ONLY JSON:
{"dataset_ids": ["id1"], "needs_clarification": false, "message": "Using X"}
OR if unclear:
{"dataset_ids": [], "needs_clarification": true, "message": "Which dataset?",
 "options": [{"dataset_id": "...", "filename": "..."}]}
Rules: match by filename/keywords. "compare"/"both"/"all" = multiple. Use exact IDs."""

    user_prompt = f"Datasets:\n{ds_list}\n\nMessage: \"{message}\""

    try:
        response = await call_llm(system_prompt, user_prompt)
        result = extract_json_from_response(response)
        valid_ids = {d["dataset_id"] for d in all_datasets}
        result["dataset_ids"] = [did for did in result.get("dataset_ids", []) if did in valid_ids]

        if not result["dataset_ids"] and not result.get("needs_clarification"):
            result["needs_clarification"] = True
            result["message"] = "Which dataset should I use?"
            result["options"] = [{"dataset_id": d["dataset_id"], "filename": d["filename"]} for d in all_datasets]

        return result
    except Exception:
        return {
            "dataset_ids": [],
            "needs_clarification": True,
            "message": "Which dataset should I use?",
            "options": [{"dataset_id": d["dataset_id"], "filename": d["filename"]} for d in all_datasets],
        }


# ─── Helpers ─────────────────────────────────────────────

def _merge_datasets(datasets: dict) -> pd.DataFrame:
    frames = []
    for did, ds in datasets.items():
        temp = ds["df"].copy()
        temp["_source_file"] = ds["meta"]["filename"]
        frames.append(temp)
    return pd.concat(frames, ignore_index=True)


def _fuzzy_match(name: str, columns: set) -> str:
    """Match column name case-insensitively or by partial match."""
    if name in columns:
        return name
    lower_map = {c.lower(): c for c in columns}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    for col in columns:
        if name.lower() in col.lower() or col.lower() in name.lower():
            return col
    return name