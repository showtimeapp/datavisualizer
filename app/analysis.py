"""
Analysis — summary stats, correlations, aggregations, and UNRESTRICTED natural language queries.
The NL query engine generates actual pandas code and executes it safely.
"""

import pandas as pd
import numpy as np
import logging
import traceback
import math

from app.llm_client import call_llm, extract_json_from_response

logger = logging.getLogger(__name__)


# ─── Pre-Built Analytics ─────────────────────────────────

def summary_statistics(df: pd.DataFrame) -> dict:
    """Full summary stats for all numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {"message": "No numeric columns found"}

    stats = numeric.describe().round(2).to_dict()
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
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    return df.nlargest(n, column).to_dict(orient="records")


def bottom_n(df: pd.DataFrame, column: str, n: int = 5) -> list[dict]:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    return df.nsmallest(n, column).to_dict(orient="records")


def group_by_agg(df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str = "sum") -> list[dict]:
    if group_col not in df.columns or agg_col not in df.columns:
        raise ValueError(f"Column not found: check '{group_col}' and '{agg_col}'")

    valid_funcs = {"sum", "mean", "median", "min", "max", "count", "std"}
    if agg_func not in valid_funcs:
        raise ValueError(f"Invalid agg_func. Must be one of: {valid_funcs}")

    result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    result.columns = [group_col, f"{agg_col}_{agg_func}"]
    return result.round(2).to_dict(orient="records")


def period_change(df: pd.DataFrame, column: str) -> list[dict]:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    result = df.copy()
    result[f"{column}_change"] = result[column].diff().round(2)
    result[f"{column}_change_pct"] = (result[column].pct_change() * 100).round(2)
    return result.to_dict(orient="records")


# ═══════════════════════════════════════════════════════════
#  Natural Language Query — CODE GENERATION + EXECUTION
# ═══════════════════════════════════════════════════════════

async def natural_language_query(df: pd.DataFrame, question: str) -> dict:
    """
    User asks ANY question in plain English.
    LLM generates pandas code → we execute it safely → return results.
    """
    # Build rich schema description
    schema_lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        sample = df[col].dropna().head(3).tolist()
        
        extra = ""
        if pd.api.types.is_numeric_dtype(df[col]):
            mn, mx = df[col].min(), df[col].max()
            extra = f", range: [{mn} to {mx}]"
        
        schema_lines.append(f"  - '{col}' (dtype: {dtype}, {nunique} unique{extra}): samples = {sample}")
    
    schema_text = "\n".join(schema_lines)

    system_prompt = """You are an expert pandas data analyst. Given a DataFrame schema and a user question,
write Python code using pandas/numpy to answer the question.

CRITICAL RULES:
1. The DataFrame is available as the variable `df`. Do NOT create or load any data.
2. Your code MUST assign the final answer to a variable called `result`.
3. `result` can be: a string, a number, a dict, a list, a DataFrame, or a Series.
4. You can create new columns, do calculations, groupby, merge, pivot, anything pandas supports.
5. Use ONLY pandas (`pd`) and numpy (`np`) — they are already imported.
6. Do NOT use: import, open, exec, eval, os, sys, subprocess, __import__, file I/O, or network calls.
7. Do NOT print anything. Just assign to `result`.
8. If the question asks to "list columns" or "show columns", just do: result = df.columns.tolist()
9. Keep the result concise — if it's a big DataFrame, limit to .head(20).
10. For calculated metrics, show your work — create intermediate columns if helpful.
11. Round numerical results to 2 decimal places.
12. If the question is ambiguous, make reasonable assumptions and note them.

ALSO generate a brief natural language `explanation` of the answer.

Return ONLY valid JSON in this format:
{
    "code": "# your pandas code here\\nresult = ...",
    "explanation": "Brief explanation of what the code does and the answer"
}

EXAMPLES:

Question: "list all columns"
{
    "code": "result = {'columns': df.columns.tolist(), 'total': len(df.columns), 'dtypes': df.dtypes.astype(str).to_dict()}",
    "explanation": "Here are all the columns in the dataset with their data types."
}

Question: "which stock had the highest daily return?"
{
    "code": "df['daily_return'] = ((df['Close'] - df['Open']) / df['Open'] * 100).round(2)\\nresult = df.nlargest(10, 'daily_return')[['Symbol', 'Date', 'Open', 'Close', 'daily_return']].to_dict(orient='records')",
    "explanation": "Calculated daily return as (Close-Open)/Open*100 and found the top 10 highest."
}

Question: "what is the average volume per sector?"
{
    "code": "result = df.groupby('Sector')['Volume'].mean().round(2).sort_values(ascending=False).reset_index().to_dict(orient='records')",
    "explanation": "Average trading volume grouped by sector, sorted highest to lowest."
}

Question: "show correlation between price and volume"
{
    "code": "numeric_cols = df.select_dtypes(include='number').columns.tolist()\\ncorr = df[numeric_cols].corr().round(3)\\nresult = corr.to_dict()",
    "explanation": "Correlation matrix between all numeric columns."
}

Question: "how many unique stocks are there?"
{
    "code": "result = {'unique_stocks': df['Symbol'].nunique(), 'stock_list': df['Symbol'].unique().tolist()[:50]}",
    "explanation": "Count of unique stock symbols in the dataset."
}"""

    user_prompt = f"""DataFrame schema ({len(df)} rows, {len(df.columns)} columns):
{schema_text}

First 5 rows:
{df.head(5).to_string()}

Question: "{question}"
"""

    try:
        response = await call_llm(system_prompt, user_prompt)
        parsed = extract_json_from_response(response)
        
        code = parsed.get("code", "")
        explanation = parsed.get("explanation", "")

        # Execute the generated code safely
        exec_result = _safe_execute(df, code)

        if exec_result["success"]:
            # Convert result to JSON-serializable format
            data = _serialize_result(exec_result["result"])
            answer = explanation or "Here's the result."

            # If result is a simple value, include it in the answer
            if isinstance(exec_result["result"], (int, float, str, bool)):
                answer = f"{explanation}\n\nAnswer: {exec_result['result']}"

            return {
                "question": question,
                "answer": answer,
                "data": data,
                "code_executed": code,
            }
        else:
            # Code execution failed — try to recover with a simpler approach
            logger.warning(f"Code execution failed: {exec_result['error']}")
            return await _fallback_query(df, question, exec_result["error"])

    except Exception as e:
        logger.error(f"NL query failed: {traceback.format_exc()}")
        return await _fallback_query(df, question, str(e))


def _safe_execute(df: pd.DataFrame, code: str) -> dict:
    """
    Execute pandas code in a restricted environment.
    Only pd, np, and the DataFrame are available.
    """
    # Security checks — block dangerous patterns
    dangerous = [
        "import ", "__import__", "exec(", "eval(", "compile(",
        "open(", "os.", "sys.", "subprocess", "shutil",
        "globals(", "locals(", "getattr(", "setattr(", "delattr(",
        "__builtins__", "__class__", "__subclasses__",
        "requests.", "urllib.", "socket.", "http.",
        "pickle.", "shelve.", "marshal.",
        "breakpoint(", "input(",
    ]
    
    code_lower = code.lower()
    for pattern in dangerous:
        if pattern.lower() in code_lower:
            return {
                "success": False,
                "result": None,
                "error": f"Blocked: code contains disallowed pattern '{pattern}'",
            }

    # Create restricted execution environment
    safe_globals = {
        "__builtins__": {
            # Allow only safe builtins
            "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter,
            "sorted": sorted, "reversed": reversed,
            "min": min, "max": max, "sum": sum,
            "abs": abs, "round": round, "pow": pow,
            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "isinstance": isinstance, "type": type,
            "True": True, "False": False, "None": None,
            "print": lambda *a, **k: None,  # no-op print
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "ZeroDivisionError": ZeroDivisionError,
        },
        "pd": pd,
        "np": np,
        "df": df.copy(),  # Work on a copy to protect the original
        "math": math,
    }

    safe_locals = {}

    try:
        exec(code, safe_globals, safe_locals)

        if "result" not in safe_locals:
            return {
                "success": False,
                "result": None,
                "error": "Code did not assign a value to 'result'",
            }

        return {
            "success": True,
            "result": safe_locals["result"],
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": f"{type(e).__name__}: {str(e)}",
        }


def _serialize_result(result) -> any:
    """Convert pandas/numpy objects to JSON-safe Python types."""
    if result is None:
        return None

    if isinstance(result, (int, float, str, bool)):
        if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
            return None
        return result

    if isinstance(result, (np.integer,)):
        return int(result)

    if isinstance(result, (np.floating,)):
        return None if np.isnan(result) else float(result)

    if isinstance(result, np.ndarray):
        return result.tolist()

    if isinstance(result, pd.DataFrame):
        # Limit size and clean NaN
        out = result.head(50)
        return out.where(out.notna(), None).to_dict(orient="records")

    if isinstance(result, pd.Series):
        out = result.head(50)
        # Try to convert to a dict with clean keys
        if result.name:
            return {str(result.name): out.where(out.notna(), None).to_dict()}
        return out.where(out.notna(), None).to_dict()

    if isinstance(result, dict):
        return _clean_dict(result)

    if isinstance(result, (list, tuple)):
        return [_serialize_result(item) for item in result]

    # Fallback — convert to string
    return str(result)


def _clean_dict(d: dict) -> dict:
    """Recursively clean a dict for JSON serialization."""
    clean = {}
    for k, v in d.items():
        key = str(k)
        if isinstance(v, dict):
            clean[key] = _clean_dict(v)
        elif isinstance(v, (list, tuple)):
            clean[key] = [_serialize_result(item) for item in v]
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[key] = None
        elif isinstance(v, (np.integer,)):
            clean[key] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[key] = None if np.isnan(v) else float(v)
        elif isinstance(v, pd.Timestamp):
            clean[key] = v.isoformat()
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            clean[key] = _serialize_result(v)
        else:
            clean[key] = v
    return clean


async def _fallback_query(df: pd.DataFrame, question: str, error: str) -> dict:
    """
    If code execution fails, try a simpler conversational approach.
    LLM answers based on the data summary instead of code.
    """
    # Build a data summary
    summary_parts = [
        f"Dataset: {len(df)} rows, {len(df.columns)} columns",
        f"Columns: {df.columns.tolist()}",
        f"Dtypes:\n{df.dtypes.to_string()}",
    ]

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        summary_parts.append(f"\nNumeric stats:\n{numeric.describe().round(2).to_string()}")

    cat_cols = df.select_dtypes(include=["object", "category"])
    if not cat_cols.empty:
        for col in cat_cols.columns[:5]:
            summary_parts.append(f"\n{col} unique values ({df[col].nunique()}): {df[col].unique()[:10].tolist()}")

    summary_parts.append(f"\nFirst 5 rows:\n{df.head(5).to_string()}")

    data_summary = "\n".join(summary_parts)

    system_prompt = """You are a financial data analyst. Given a dataset summary and a question,
provide a helpful answer based on what you can see in the data.

If you can compute the answer from the summary/stats provided, do so.
If you cannot answer precisely, explain what would be needed and give your best estimate.
Be specific, use numbers from the data, and be concise."""

    user_prompt = f"""Previous code execution failed with: {error}

Data summary:
{data_summary}

Question: "{question}"

Please answer based on the data summary above."""

    try:
        response = await call_llm(system_prompt, user_prompt)
        return {
            "question": question,
            "answer": response,
            "data": {
                "note": "Answered from data summary (code execution was not needed or failed)",
                "columns": df.columns.tolist(),
                "shape": list(df.shape),
            },
        }
    except Exception as e:
        return {
            "question": question,
            "answer": f"I couldn't process this question. Error: {str(e)}. Try rephrasing or asking about specific columns: {df.columns.tolist()}",
            "data": {"columns": df.columns.tolist()},
        }