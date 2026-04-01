"""
Chart generation — builds Plotly JSON specs for all supported chart types.
Includes auto-suggestion based on data shape.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json


# ─── Auto-Suggest ────────────────────────────────────────

def suggest_charts(df: pd.DataFrame) -> list[dict]:
    """Analyze data shape and recommend chart types with column mappings."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Also check if any object columns look like dates
    for col in cat_cols[:]:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(df[col].head(5), errors="coerce")
            if parsed.notna().sum() >= 3:
                date_cols.append(col)
                cat_cols.remove(col)
        except Exception:
            pass

    suggestions = []

    # Time series
    if date_cols and numeric_cols:
        x = date_cols[0]
        y = numeric_cols[:3]
        suggestions.append({
            "chart_type": "line",
            "x": x, "y": y,
            "reason": f"Time series data: {x} vs {', '.join(y)}"
        })
        suggestions.append({
            "chart_type": "area",
            "x": x, "y": y,
            "reason": f"Area view of trends over {x}"
        })

    # Category + measures
    if cat_cols and numeric_cols:
        x = cat_cols[0]
        y = numeric_cols[:2]
        uniques = df[x].nunique()

        suggestions.append({
            "chart_type": "bar",
            "x": x, "y": y,
            "reason": f"Compare {', '.join(y)} across {x}"
        })

        if uniques <= 10 and len(numeric_cols) >= 1:
            suggestions.append({
                "chart_type": "pie",
                "x": x, "y": [numeric_cols[0]],
                "reason": f"Distribution of {numeric_cols[0]} by {x}"
            })

        if uniques <= 8 and len(numeric_cols) >= 3:
            suggestions.append({
                "chart_type": "radar",
                "x": x, "y": numeric_cols[:5],
                "reason": f"Multi-metric comparison across {x}"
            })

    # Two numeric columns
    if len(numeric_cols) >= 2:
        suggestions.append({
            "chart_type": "scatter",
            "x": numeric_cols[0], "y": [numeric_cols[1]],
            "reason": f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}"
        })

    # Single numeric → histogram
    if numeric_cols:
        suggestions.append({
            "chart_type": "histogram",
            "x": numeric_cols[0], "y": [],
            "reason": f"Distribution of {numeric_cols[0]}"
        })

    # Multiple numeric → heatmap
    if len(numeric_cols) >= 3:
        suggestions.append({
            "chart_type": "heatmap",
            "x": "", "y": numeric_cols,
            "reason": "Correlation heatmap of all numeric columns"
        })

    # Box plot
    if cat_cols and numeric_cols:
        suggestions.append({
            "chart_type": "box",
            "x": cat_cols[0], "y": [numeric_cols[0]],
            "reason": f"Spread of {numeric_cols[0]} across {cat_cols[0]}"
        })

    return suggestions[:8]  # Cap at 8 suggestions


# ─── Chart Builders ──────────────────────────────────────

def build_chart(df: pd.DataFrame, chart_type: str, x: str, y: list[str],
                color: str = None, title: str = None) -> dict:
    """Build a Plotly chart and return its JSON spec."""

    builder = CHART_BUILDERS.get(chart_type)
    if not builder:
        raise ValueError(f"Unknown chart type: {chart_type}. Supported: {list(CHART_BUILDERS.keys())}")

    fig = builder(df, x, y, color, title)

    # Apply consistent theme
    fig.update_layout(
        template="plotly_white",
        title=title or f"{chart_type.title()} Chart",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=12),
    )

    return json.loads(fig.to_json())


def _build_bar(df, x, y, color, title):
    if len(y) == 1:
        return px.bar(df, x=x, y=y[0], color=color, title=title)
    fig = go.Figure()
    for col in y:
        fig.add_trace(go.Bar(name=col, x=df[x], y=df[col]))
    fig.update_layout(barmode="group")
    return fig


def _build_line(df, x, y, color, title):
    if len(y) == 1:
        return px.line(df, x=x, y=y[0], color=color, title=title, markers=True)
    fig = go.Figure()
    for col in y:
        fig.add_trace(go.Scatter(name=col, x=df[x], y=df[col], mode="lines+markers"))
    return fig


def _build_area(df, x, y, color, title):
    fig = go.Figure()
    for col in y:
        fig.add_trace(go.Scatter(name=col, x=df[x], y=df[col], fill="tonexty", mode="lines"))
    return fig


def _build_pie(df, x, y, color, title):
    return px.pie(df, names=x, values=y[0], title=title)


def _build_donut(df, x, y, color, title):
    fig = go.Figure(go.Pie(labels=df[x], values=df[y[0]], hole=0.4))
    return fig


def _build_scatter(df, x, y, color, title):
    return px.scatter(df, x=x, y=y[0], color=color, title=title)


def _build_histogram(df, x, y, color, title):
    return px.histogram(df, x=x, color=color, title=title)


def _build_heatmap(df, x, y, color, title):
    numeric = df[y] if y else df.select_dtypes(include="number")
    corr = numeric.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu_r", zmid=0, text=corr.values, texttemplate="%{text}",
    ))
    return fig


def _build_box(df, x, y, color, title):
    return px.box(df, x=x, y=y[0], color=color, title=title)


def _build_waterfall(df, x, y, color, title):
    values = df[y[0]].tolist()
    measures = ["relative"] * len(values)
    measures[0] = "absolute"
    fig = go.Figure(go.Waterfall(
        x=df[x].tolist(), y=values, measure=measures,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    return fig


def _build_funnel(df, x, y, color, title):
    return px.funnel(df, x=y[0], y=x, title=title)


def _build_radar(df, x, y, color, title):
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[col] for col in y],
            theta=y,
            fill="toself",
            name=str(row[x]) if x else "",
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
    return fig


def _build_candlestick(df, x, y, color, title):
    """Expects y = [open, high, low, close]."""
    if len(y) < 4:
        raise ValueError("Candlestick needs y = [open, high, low, close]")
    fig = go.Figure(go.Candlestick(
        x=df[x], open=df[y[0]], high=df[y[1]], low=df[y[2]], close=df[y[3]]
    ))
    return fig


def _build_treemap(df, x, y, color, title):
    return px.treemap(df, path=[x], values=y[0], title=title)


# ─── Registry ────────────────────────────────────────────

CHART_BUILDERS = {
    "bar": _build_bar,
    "line": _build_line,
    "area": _build_area,
    "pie": _build_pie,
    "donut": _build_donut,
    "scatter": _build_scatter,
    "histogram": _build_histogram,
    "heatmap": _build_heatmap,
    "box": _build_box,
    "waterfall": _build_waterfall,
    "funnel": _build_funnel,
    "radar": _build_radar,
    "candlestick": _build_candlestick,
    "treemap": _build_treemap,
}


# ─── Smart Chart (NL → Chart Config) ────────────────────

from app.llm_client import call_llm, extract_json_from_response


async def interpret_chart_request(df: pd.DataFrame, prompt: str) -> dict:
    """
    Takes a natural language description and the dataset,
    returns a chart config: {chart_type, x, y, color, title, filters}.
    """
    # Build column info for the LLM
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        samples = df[col].dropna().head(4).tolist()
        col_info.append(f"  - '{col}' (type: {dtype}, {nunique} unique values, samples: {samples})")

    columns_desc = "\n".join(col_info)

    system_prompt = f"""You are a chart configuration engine for a financial data visualization tool.

Given a user's natural language description of what chart they want, and the dataset's column info,
return a JSON object with the chart configuration.

Available chart types: bar, line, area, pie, donut, scatter, histogram, heatmap, box, waterfall, funnel, radar, candlestick, treemap

Rules:
- Return ONLY valid JSON, no markdown, no explanation.
- You MUST pick column names EXACTLY as they appear in the dataset — spelling, case, spaces must match.
- If the user doesn't specify a chart type, pick the best one for their description.
- If the user mentions filtering (e.g., "only for Q1" or "just technology sector"), include a filters array.
- If the user says something vague like "show me the data" or "visualize this", pick the most insightful chart.
- For time-based data, prefer line charts. For comparisons, prefer bar. For distribution, prefer pie/donut.
- Always generate a descriptive title.

Return format:
{{
    "chart_type": "bar",
    "x": "column_name_for_x_axis",
    "y": ["column_name_1", "column_name_2"],
    "color": "optional_color_column_or_null",
    "title": "Descriptive Chart Title",
    "filters": [
        {{"column": "col_name", "condition": "eq|gt|lt|gte|lte|contains", "value": "..."}}
    ]
}}

Notes on y field:
- For pie/donut/histogram/treemap: y should be a single-element list with the values column
- For scatter: y is a single-element list with the y-axis column
- For heatmap: y should list all numeric columns to correlate
- For candlestick: y must be exactly [open, high, low, close] columns
- For all others: y can be one or more measure columns
- If the user asks for something impossible with the data, pick the closest feasible chart and note it in the title."""

    user_prompt = f"""Dataset columns:
{columns_desc}

First 3 rows:
{df.head(3).to_string()}

User request: "{prompt}"
"""

    response = await call_llm(system_prompt, user_prompt)
    config = extract_json_from_response(response)

    # Validate column names exist
    all_cols = set(df.columns.tolist())

    if config.get("x") and config["x"] not in all_cols:
        # Try case-insensitive match
        config["x"] = _fuzzy_match_column(config["x"], all_cols)

    if config.get("y"):
        config["y"] = [_fuzzy_match_column(c, all_cols) for c in config["y"]]

    if config.get("color") and config["color"] not in all_cols:
        config["color"] = None  # Drop invalid color column

    # Validate chart type
    if config.get("chart_type") not in CHART_BUILDERS:
        config["chart_type"] = "bar"  # Safe fallback

    return config


def _fuzzy_match_column(name: str, columns: set) -> str:
    """Try to match a column name case-insensitively or by partial match."""
    # Exact match
    if name in columns:
        return name

    # Case-insensitive
    lower_map = {c.lower(): c for c in columns}
    if name.lower() in lower_map:
        return lower_map[name.lower()]

    # Partial match (column contains the name or vice versa)
    for col in columns:
        if name.lower() in col.lower() or col.lower() in name.lower():
            return col

    # Return original — will fail at chart build time with a clear error
    return name


async def smart_chart(df: pd.DataFrame, prompt: str) -> tuple[dict, pd.DataFrame]:
    """
    Full pipeline: interpret prompt → apply filters → build chart.
    Returns (plotly_json_spec, chart_config).
    """
    config = await interpret_chart_request(df, prompt)

    # Apply filters if any
    filtered_df = df.copy()
    filters = config.get("filters", [])
    for f in filters:
        col = f.get("column", "")
        if col not in filtered_df.columns:
            col = _fuzzy_match_column(col, set(filtered_df.columns))
        if col not in filtered_df.columns:
            continue

        cond = f.get("condition", "eq")
        val = f.get("value")

        try:
            if cond == "eq":
                filtered_df = filtered_df[filtered_df[col] == val]
            elif cond == "contains":
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(val), case=False, na=False)]
            elif cond in ("gt", "lt", "gte", "lte"):
                numeric_val = float(val)
                ops = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}
                filtered_df = filtered_df.query(f"`{col}` {ops[cond]} @numeric_val")
        except Exception:
            continue  # Skip invalid filters gracefully

    if filtered_df.empty:
        filtered_df = df  # Fall back to unfiltered if filters removed everything

    # Build the chart
    chart_type = config["chart_type"]
    x = config.get("x", "")
    y = config.get("y", [])
    color = config.get("color")
    title = config.get("title", f"{chart_type.title()} Chart")

    plotly_json = build_chart(filtered_df, chart_type, x, y, color, title)

    return plotly_json, config