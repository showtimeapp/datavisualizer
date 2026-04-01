"""
Brain / Router — LLM decides: is this a chart, analysis, query, or needs clarification?
Single entry point for all user messages.
"""

import logging
import pandas as pd

from app.llm_client import call_llm, extract_json_from_response
from app.storage import load_dataset, load_metadata, list_datasets
from app.analysis import (
    summary_statistics, correlation_matrix,
    top_n, bottom_n, group_by_agg, period_change,
    natural_language_query,
)
from app.charts import smart_chart, suggest_charts

logger = logging.getLogger(__name__)


async def process_message(message: str, dataset_ids: list[str] | None = None) -> dict:
    """
    Main brain — takes a user message, figures out what to do, and returns a response.
    
    Returns dict with keys: intent, message, chart, analysis, data, options, datasets_used, etc.
    """
    # Step 1: Get available datasets
    all_datasets = list_datasets()
    if not all_datasets:
        return {
            "intent": "clarify",
            "message": "No datasets uploaded yet. Please upload a file first.",
            "options": None,
            "datasets_used": [],
        }

    # Step 2: Resolve which dataset(s) to use
    if dataset_ids:
        # User explicitly specified
        selected_ids = dataset_ids
    elif len(all_datasets) == 1:
        # Only one dataset — use it automatically
        selected_ids = [all_datasets[0]["dataset_id"]]
    else:
        # Multiple datasets — ask LLM which one(s) the user means
        resolved = await _resolve_datasets(message, all_datasets)
        if resolved["needs_clarification"]:
            return {
                "intent": "clarify",
                "message": resolved["message"],
                "options": resolved["options"],
                "datasets_used": [],
            }
        selected_ids = resolved["dataset_ids"]

    # Step 3: Load the datasets
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
                "message": f"Dataset '{did}' not found. Please check the ID.",
                "options": [{"dataset_id": d["dataset_id"], "filename": d["filename"]}
                           for d in all_datasets],
                "datasets_used": [],
            }

    # Step 4: Determine intent and execute
    intent = await _classify_intent(message, datasets)

    if intent["type"] == "chart":
        return await _handle_chart(message, datasets, selected_ids)
    elif intent["type"] == "analysis":
        return await _handle_analysis(message, datasets, selected_ids, intent)
    elif intent["type"] == "query":
        return await _handle_query(message, datasets, selected_ids)
    else:
        return await _handle_query(message, datasets, selected_ids)


# ─── Intent Classification ───────────────────────────────

async def _classify_intent(message: str, datasets: dict) -> dict:
    """Use LLM to classify what the user wants."""
    # Build dataset descriptions
    ds_desc = []
    for did, ds in datasets.items():
        meta = ds["meta"]
        cols = [c["name"] for c in meta["columns"]]
        ds_desc.append(f"  - {meta['filename']} (id: {did}): columns = {cols}, {meta['row_count']} rows")
    ds_text = "\n".join(ds_desc)

    system_prompt = """You are an intent classifier for a financial data tool.
Given a user message and dataset info, classify the intent.

Return ONLY valid JSON:
{
    "type": "chart" | "analysis" | "query",
    "analysis_subtype": "summary" | "correlation" | "top_n" | "bottom_n" | "group_by" | "period_change" | null,
    "analysis_params": { "column": "...", "n": 5, "group_column": "...", "agg_column": "...", "agg_func": "sum" } or null,
    "reason": "brief explanation"
}

Guidelines:
- "chart" = user wants to SEE/VISUALIZE data (mentions chart, graph, plot, show, visualize, compare visually, trend, pie, bar, line, scatter, etc.)
- "analysis" = user wants NUMBERS/STATS (summary, statistics, average, correlation, top 5, group by, percentage change, etc.)
- "query" = user asks a QUESTION about the data (what was the highest X, which Y had, when did Z, how many, etc.)

If ambiguous, prefer "query" — it's the most flexible."""

    user_prompt = f"Datasets:\n{ds_text}\n\nUser message: \"{message}\""

    try:
        response = await call_llm(system_prompt, user_prompt)
        return extract_json_from_response(response)
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}, defaulting to query")
        return {"type": "query", "analysis_subtype": None, "analysis_params": None}


# ─── Dataset Resolution ──────────────────────────────────

async def _resolve_datasets(message: str, all_datasets: list[dict]) -> dict:
    """Figure out which dataset(s) the user is referring to."""
    ds_list = []
    for d in all_datasets:
        ds_list.append(f"  - id: \"{d['dataset_id']}\", filename: \"{d['filename']}\", rows: {d['row_count']}")
    ds_text = "\n".join(ds_list)

    system_prompt = """You are a dataset resolver for a financial data tool.
The user has uploaded multiple datasets. Given their message, decide which dataset(s) they need.

Return ONLY valid JSON:
{
    "dataset_ids": ["id1", "id2"],
    "needs_clarification": false,
    "message": "Using sales_data.csv"
}

OR if you can't tell which dataset(s):
{
    "dataset_ids": [],
    "needs_clarification": true,
    "message": "Which dataset would you like me to use for this?",
    "options": [{"dataset_id": "...", "filename": "...", "reason": "might be relevant because..."}]
}

Rules:
- If the message mentions a filename or keywords matching a dataset, pick it.
- If the message says "compare" or "both" or "all", pick multiple.
- If the message is generic (like "show me a chart"), and datasets are about different topics, ask.
- If datasets have similar content, pick the most relevant one.
- ALWAYS use exact dataset_id values from the list provided."""

    user_prompt = f"Available datasets:\n{ds_text}\n\nUser message: \"{message}\""

    try:
        response = await call_llm(system_prompt, user_prompt)
        result = extract_json_from_response(response)

        # Validate returned IDs
        valid_ids = {d["dataset_id"] for d in all_datasets}
        result["dataset_ids"] = [did for did in result.get("dataset_ids", []) if did in valid_ids]

        if not result["dataset_ids"] and not result.get("needs_clarification"):
            result["needs_clarification"] = True
            result["message"] = "I'm not sure which dataset to use. Could you pick one?"
            result["options"] = [
                {"dataset_id": d["dataset_id"], "filename": d["filename"]}
                for d in all_datasets
            ]

        return result
    except Exception as e:
        logger.warning(f"Dataset resolution failed: {e}")
        return {
            "dataset_ids": [],
            "needs_clarification": True,
            "message": "I couldn't determine which dataset to use. Please select one:",
            "options": [
                {"dataset_id": d["dataset_id"], "filename": d["filename"]}
                for d in all_datasets
            ],
        }


# ─── Handlers ────────────────────────────────────────────

async def _handle_chart(message: str, datasets: dict, dataset_ids: list[str]) -> dict:
    """Generate a chart from natural language."""
    # If multiple datasets, merge them for the chart
    if len(datasets) > 1:
        df = _merge_datasets(datasets)
        source_note = f"Merged from: {', '.join(ds['meta']['filename'] for ds in datasets.values())}"
    else:
        did = list(datasets.keys())[0]
        df = datasets[did]["df"]
        source_note = datasets[did]["meta"]["filename"]

    try:
        plotly_json, config = await smart_chart(df, message)

        explanation = _build_chart_explanation(config, source_note)

        return {
            "intent": "chart",
            "message": explanation,
            "chart": plotly_json,
            "chart_type": config.get("chart_type"),
            "chart_config": {
                "x": config.get("x"),
                "y": config.get("y"),
                "color": config.get("color"),
                "title": config.get("title"),
                "filters_applied": config.get("filters", []),
            },
            "datasets_used": dataset_ids,
        }
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        return {
            "intent": "chart",
            "message": f"I couldn't generate that chart: {str(e)}. Try rephrasing — e.g., specify which columns to use.",
            "chart": None,
            "datasets_used": dataset_ids,
        }


async def _handle_analysis(message: str, datasets: dict, dataset_ids: list[str], intent: dict) -> dict:
    """Run pre-built analytics."""
    did = list(datasets.keys())[0]
    df = datasets[did]["df"]
    meta = datasets[did]["meta"]

    subtype = intent.get("analysis_subtype", "summary")
    params = intent.get("analysis_params") or {}

    try:
        if subtype == "summary":
            result = summary_statistics(df)
            explanation = f"Here are the summary statistics for {meta['filename']}."

        elif subtype == "correlation":
            result = correlation_matrix(df)
            explanation = f"Correlation matrix for numeric columns in {meta['filename']}."

        elif subtype == "top_n":
            col = params.get("column") or _first_numeric_col(df)
            n = params.get("n", 5)
            result = top_n(df, col, n)
            explanation = f"Top {n} rows by {col}."

        elif subtype == "bottom_n":
            col = params.get("column") or _first_numeric_col(df)
            n = params.get("n", 5)
            result = bottom_n(df, col, n)
            explanation = f"Bottom {n} rows by {col}."

        elif subtype == "group_by":
            gcol = params.get("group_column") or _first_cat_col(df)
            acol = params.get("agg_column") or _first_numeric_col(df)
            func = params.get("agg_func", "sum")
            result = group_by_agg(df, gcol, acol, func)
            explanation = f"{func.title()} of {acol} grouped by {gcol}."

        elif subtype == "period_change":
            col = params.get("column") or _first_numeric_col(df)
            result = period_change(df, col)
            explanation = f"Period-over-period change for {col}."

        else:
            result = summary_statistics(df)
            explanation = f"Summary statistics for {meta['filename']}."

        # For multi-file analysis
        if len(datasets) > 1:
            results = {}
            for did2, ds in datasets.items():
                results[ds["meta"]["filename"]] = summary_statistics(ds["df"])
            result = results
            explanation = f"Comparative analysis across {len(datasets)} datasets."

        return {
            "intent": "analysis",
            "message": explanation,
            "analysis": result,
            "datasets_used": dataset_ids,
        }

    except Exception as e:
        return {
            "intent": "analysis",
            "message": f"Analysis failed: {str(e)}",
            "analysis": None,
            "datasets_used": dataset_ids,
        }


async def _handle_query(message: str, datasets: dict, dataset_ids: list[str]) -> dict:
    """Answer a question about the data."""
    # For single dataset
    if len(datasets) == 1:
        did = list(datasets.keys())[0]
        df = datasets[did]["df"]
        meta = datasets[did]["meta"]

        result = await natural_language_query(df, message)

        return {
            "intent": "query",
            "message": result.get("answer", "I couldn't find an answer."),
            "data": result.get("data"),
            "datasets_used": dataset_ids,
        }

    # For multiple datasets — query each and synthesize
    all_results = []
    for did, ds in datasets.items():
        result = await natural_language_query(ds["df"], message)
        all_results.append({
            "filename": ds["meta"]["filename"],
            "answer": result.get("answer"),
            "data": result.get("data"),
        })

    combined_answer = "\n".join(
        f"**{r['filename']}**: {r['answer']}" for r in all_results
    )

    return {
        "intent": "query",
        "message": combined_answer,
        "data": all_results,
        "datasets_used": dataset_ids,
    }


# ─── Helpers ─────────────────────────────────────────────

def _merge_datasets(datasets: dict) -> pd.DataFrame:
    """Merge multiple DataFrames — smart concat with source column."""
    frames = []
    for did, ds in datasets.items():
        temp = ds["df"].copy()
        temp["_source_file"] = ds["meta"]["filename"]
        frames.append(temp)
    return pd.concat(frames, ignore_index=True)


def _first_numeric_col(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) == 0:
        raise ValueError("No numeric columns found")
    return numeric[0]


def _first_cat_col(df: pd.DataFrame) -> str:
    cats = df.select_dtypes(include=["object", "category"]).columns
    if len(cats) == 0:
        raise ValueError("No categorical columns found")
    return cats[0]


def _build_chart_explanation(config: dict, source: str) -> str:
    """Build a human-readable explanation of what chart was generated."""
    ct = config.get("chart_type", "chart")
    title = config.get("title", "")
    x = config.get("x", "")
    y = config.get("y", [])
    filters = config.get("filters", [])

    parts = [f"Generated a **{ct}** chart"]
    if title:
        parts[0] = f"**{title}**"
    if x and y:
        parts.append(f"showing {', '.join(y)} on the Y-axis against {x} on the X-axis")
    if filters:
        filter_desc = ", ".join(f"{f['column']} {f['condition']} {f['value']}" for f in filters)
        parts.append(f"filtered by: {filter_desc}")
    parts.append(f"(source: {source})")

    return " — ".join(parts) + "."