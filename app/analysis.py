"""
Analysis — summary stats, correlations, aggregations, and natural language queries.
"""

import pandas as pd
import numpy as np

from app.llm_client import call_llm, extract_json_from_response


# ─── Pre-Built Analytics ─────────────────────────────────

def summary_statistics(df: pd.DataFrame) -> dict:
    """Full summary stats for all numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {"message": "No numeric columns found"}

    stats = numeric.describe().round(2).to_dict()
    # Add additional useful metrics
    for col in numeric.columns:
        stats[col]["missing"] = int(df[col].isna().sum())
        stats[col]["missing_pct"] = round(df[col].isna().mean() * 100, 1)
        if numeric[col].dropna().shape[0] > 0:
            stats[col]["median"] = round(numeric[col].median(), 2)
            stats[col]["skew"] = round(numeric[col].skew(), 2)
    return stats


def correlation_matrix(df: pd.DataFrame) -> dict:
    """Pearson correlation between all numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"message": "Need at least 2 numeric columns for correlation"}
    corr = numeric.corr().round(3)
    return corr.to_dict()


def top_n(df: pd.DataFrame, column: str, n: int = 5) -> list[dict]:
    """Top N rows by a column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    return df.nlargest(n, column).to_dict(orient="records")


def bottom_n(df: pd.DataFrame, column: str, n: int = 5) -> list[dict]:
    """Bottom N rows by a column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    return df.nsmallest(n, column).to_dict(orient="records")


def group_by_agg(df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str = "sum") -> list[dict]:
    """Group by a column and aggregate another."""
    if group_col not in df.columns or agg_col not in df.columns:
        raise ValueError(f"Column not found: check '{group_col}' and '{agg_col}'")

    valid_funcs = {"sum", "mean", "median", "min", "max", "count", "std"}
    if agg_func not in valid_funcs:
        raise ValueError(f"Invalid agg_func. Must be one of: {valid_funcs}")

    result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    result.columns = [group_col, f"{agg_col}_{agg_func}"]
    return result.round(2).to_dict(orient="records")


def period_change(df: pd.DataFrame, column: str) -> list[dict]:
    """Calculate row-over-row absolute and percentage change."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    result = df.copy()
    result[f"{column}_change"] = result[column].diff().round(2)
    result[f"{column}_change_pct"] = (result[column].pct_change() * 100).round(2)
    return result.to_dict(orient="records")


# ─── Natural Language Query (LLM-powered) ────────────────

async def natural_language_query(df: pd.DataFrame, question: str) -> dict:
    """
    User asks a question in plain English.
    LLM returns a safe pandas operation spec → we execute it.
    """
    # Build schema description
    schema_lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(3).tolist()
        schema_lines.append(f"  - {col} ({dtype}): e.g. {sample}")
    schema_text = "\n".join(schema_lines)

    system_prompt = """You are a data analysis assistant. Given a DataFrame schema and a question,
return a JSON object describing the operation to perform.

Supported operations:
1. {"op": "filter", "column": "X", "condition": "gt|lt|eq|gte|lte|contains", "value": ...}
2. {"op": "sort", "column": "X", "ascending": true/false, "head": 5}
3. {"op": "agg", "group_by": "X" (optional), "column": "Y", "func": "sum|mean|max|min|count|median"}
4. {"op": "value_at", "column": "X", "func": "max|min|first|last"}
5. {"op": "describe", "column": "X"}

Return ONLY the JSON object, nothing else. Pick the simplest operation that answers the question.
If the question cannot be answered with these operations, return: {"op": "unsupported", "reason": "..."}
"""

    user_prompt = f"Schema:\n{schema_text}\n\nFirst 3 rows:\n{df.head(3).to_string()}\n\nQuestion: {question}"

    response = await call_llm(system_prompt, user_prompt)
    operation = extract_json_from_response(response)

    # Execute the operation safely
    return _execute_operation(df, operation, question)


def _execute_operation(df: pd.DataFrame, op: dict, question: str) -> dict:
    """Execute a validated operation spec against the DataFrame."""
    op_type = op.get("op")

    if op_type == "unsupported":
        return {"question": question, "answer": op.get("reason", "Cannot answer this question"), "data": None}

    if op_type == "filter":
        col, cond, val = op["column"], op["condition"], op["value"]
        mask = {
            "gt": df[col] > val, "lt": df[col] < val,
            "eq": df[col] == val, "gte": df[col] >= val,
            "lte": df[col] <= val,
            "contains": df[col].astype(str).str.contains(str(val), case=False, na=False),
        }.get(cond)
        if mask is None:
            return {"question": question, "answer": "Unknown filter condition", "data": None}
        result = df[mask]
        return {
            "question": question,
            "answer": f"Found {len(result)} rows where {col} {cond} {val}",
            "data": result.head(20).to_dict(orient="records"),
        }

    elif op_type == "sort":
        col = op["column"]
        asc = op.get("ascending", False)
        head = op.get("head", 5)
        result = df.sort_values(col, ascending=asc).head(head)
        direction = "ascending" if asc else "descending"
        return {
            "question": question,
            "answer": f"Top {head} rows sorted by {col} ({direction})",
            "data": result.to_dict(orient="records"),
        }

    elif op_type == "agg":
        col = op["column"]
        func = op["func"]
        group = op.get("group_by")
        if group:
            result = df.groupby(group)[col].agg(func).reset_index()
            return {
                "question": question,
                "answer": f"{func} of {col} grouped by {group}",
                "data": result.round(2).to_dict(orient="records"),
            }
        else:
            value = df[col].agg(func)
            return {
                "question": question,
                "answer": f"The {func} of {col} is {round(value, 2) if isinstance(value, float) else value}",
                "data": None,
            }

    elif op_type == "value_at":
        col = op["column"]
        func = op["func"]
        if func == "max":
            row = df.loc[df[col].idxmax()]
        elif func == "min":
            row = df.loc[df[col].idxmin()]
        elif func == "first":
            row = df.iloc[0]
        elif func == "last":
            row = df.iloc[-1]
        else:
            return {"question": question, "answer": f"Unknown func: {func}", "data": None}
        return {
            "question": question,
            "answer": f"Row with {func} {col}: {row[col]}",
            "data": row.to_dict(),
        }

    elif op_type == "describe":
        col = op["column"]
        desc = df[col].describe().round(2).to_dict()
        return {
            "question": question,
            "answer": f"Description of {col}",
            "data": desc,
        }

    return {"question": question, "answer": "Unknown operation", "data": None}
