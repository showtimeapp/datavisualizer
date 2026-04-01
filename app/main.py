"""
Financial Data Visualization Backend — Unified API
====================================================
3 endpoints:
  POST /api/upload        — Upload one or more files
  POST /api/chat          — Ask anything (chart, analysis, query) — LLM decides
  GET  /api/datasets      — List uploaded datasets
  DELETE /api/dataset/{id} — Delete a dataset
  GET  /api/playground/{id} — Interactive chart viewer (browser)
"""

import traceback
import logging
import json
import math
from datetime import datetime, date
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from app.config import settings
from app.models import (
    ChatRequest, ChatResponse, UploadResponse,
    DatasetInfo, ColumnInfo, IntentType,
)
from app.ingestion import ingest_file
from app.processing import process_dataframe
from app.storage import (
    save_dataset, load_dataset, load_metadata,
    list_datasets, delete_dataset,
)
from app.router import process_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Custom JSON Response ────────────────────────────────

class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, default=self._default, ensure_ascii=False, allow_nan=False).encode("utf-8")

    @staticmethod
    def _default(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ─── App Setup ───────────────────────────────────────────

app = FastAPI(
    title="Financial Data Visualization API",
    description=(
        "Upload financial data → ask questions, get charts, run analysis. "
        "One chat endpoint handles everything. LLM decides the intent."
    ),
    version="2.0.0",
    default_response_class=SafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_preview(df, n=5) -> list[dict]:
    records = df.head(n).to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[k] = None
            elif hasattr(v, "isoformat"):
                row[k] = v.isoformat()
    return records


# ─── Health Check ────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Financial Data Visualization API",
        "version": "2.0.0",
        "llm_provider": settings.LLM_PROVIDER,
        "endpoints": {
            "POST /api/upload": "Upload one or more files (multipart)",
            "POST /api/chat": "Ask anything — LLM routes to chart/analysis/query",
            "GET  /api/datasets": "List all uploaded datasets",
            "GET  /api/dataset/{id}": "Get dataset details + preview",
            "DELETE /api/dataset/{id}": "Delete a dataset",
            "GET  /api/playground/{id}": "Interactive chart viewer (open in browser)",
        },
    }


# ═══════════════════════════════════════════════════════════
#  1. UPLOAD — supports multiple files at once
# ═══════════════════════════════════════════════════════════

@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more files. Each file becomes a separate dataset.
    
    Supported formats: CSV, Excel (.xlsx/.xls), HTML, Markdown, JSON, TXT
    
    Example (curl):
        curl -X POST http://localhost:8000/api/upload \\
          -F "files=@sales.csv" \\
          -F "files=@expenses.xlsx" \\
          -F "files=@report.html"
    """
    results = []
    errors = []
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    for file in files:
        try:
            contents = await file.read()
            if len(contents) > max_bytes:
                errors.append(f"{file.filename}: File too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)")
                continue

            df = await ingest_file(contents, file.filename)
            df = process_dataframe(df)
            dataset_id = save_dataset(df, file.filename)
            meta = load_metadata(dataset_id)

            results.append(DatasetInfo(
                dataset_id=dataset_id,
                filename=file.filename,
                columns=[ColumnInfo(**c) for c in meta["columns"]],
                row_count=meta["row_count"],
                preview=safe_preview(df, 5),
            ))

        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {traceback.format_exc()}")
            errors.append(f"{file.filename}: {str(e)}")

    if not results and errors:
        raise HTTPException(400, detail={"message": "All uploads failed", "errors": errors})

    msg_parts = [f"Successfully uploaded {len(results)} file(s)."]
    if errors:
        msg_parts.append(f"Failed: {'; '.join(errors)}")

    return UploadResponse(
        datasets=results,
        message=" ".join(msg_parts),
    )


# ═══════════════════════════════════════════════════════════
#  2. CHAT — the single smart endpoint
# ═══════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Send any message — the system automatically decides whether to:
    - Generate a chart (bar, line, pie, scatter, etc.)
    - Run analysis (summary stats, correlation, top-N, etc.)
    - Answer a question about the data
    - Ask for clarification (which file to use)

    Examples:
        {"message": "show me a bar chart of revenue by quarter"}
        {"message": "what was the highest revenue month?"}
        {"message": "give me summary statistics"}
        {"message": "compare revenue across both files", "dataset_ids": ["abc123", "def456"]}
        {"message": "pie chart from sales.csv"}
    """
    try:
        result = await process_message(req.message, req.dataset_ids)

        return ChatResponse(
            intent=IntentType(result.get("intent", "query")),
            message=result.get("message", ""),
            chart=result.get("chart"),
            chart_type=result.get("chart_type"),
            chart_config=result.get("chart_config"),
            analysis=result.get("analysis"),
            data=result.get("data"),
            options=result.get("options"),
            datasets_used=result.get("datasets_used", []),
        )

    except Exception as e:
        logger.error(f"Chat failed: {traceback.format_exc()}")
        raise HTTPException(500, f"Processing failed: {str(e)}")


# ═══════════════════════════════════════════════════════════
#  3. DATASET MANAGEMENT
# ═══════════════════════════════════════════════════════════

@app.get("/api/datasets")
async def get_datasets():
    """List all uploaded datasets."""
    return list_datasets()


@app.get("/api/dataset/{dataset_id}")
async def get_dataset(dataset_id: str, rows: int = Query(default=10, le=100)):
    """Get dataset metadata + preview rows."""
    try:
        meta = load_metadata(dataset_id)
        df = load_dataset(dataset_id)
        meta["preview"] = safe_preview(df, rows)
        return meta
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")


@app.delete("/api/dataset/{dataset_id}")
async def remove_dataset(dataset_id: str):
    """Delete a dataset."""
    if delete_dataset(dataset_id):
        return {"deleted": dataset_id}
    raise HTTPException(404, f"Dataset '{dataset_id}' not found")


# ═══════════════════════════════════════════════════════════
#  4. PLAYGROUND — Interactive Chart Viewer (Browser)
# ═══════════════════════════════════════════════════════════

@app.get("/api/playground/{dataset_id}", response_class=HTMLResponse)
async def playground(dataset_id: str):
    """
    Interactive chart playground — open in browser.
    Type prompts, see charts rendered live.
    
    URL: http://localhost:8000/api/playground/{dataset_id}
    """
    try:
        meta = load_metadata(dataset_id)
        load_dataset(dataset_id)  # validate it exists
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")

    columns = [c["name"] for c in meta["columns"]]

    return _build_playground_html(dataset_id, meta, columns)


@app.get("/api/playground", response_class=HTMLResponse)
async def playground_all():
    """
    Interactive playground for all datasets.
    URL: http://localhost:8000/api/playground
    """
    all_datasets = list_datasets()
    if not all_datasets:
        return HTMLResponse("<h1>No datasets uploaded yet</h1><p>Upload files first via POST /api/upload</p>")
    return _build_playground_html(None, None, None, all_datasets)


def _build_playground_html(dataset_id, meta, columns, all_datasets=None):
    """Build the interactive chart playground HTML."""

    if dataset_id:
        title = f"Playground — {meta['filename']}"
        subtitle = f"{meta['filename']} — {meta['row_count']} rows"
        cols_html = " ".join(f'<span class="col-tag">{c}</span>' for c in columns)
        ds_selector = f'<input type="hidden" id="dsInput" value="{dataset_id}">'
        ds_info = f'<div class="columns-bar">Columns: {cols_html}</div>'
    else:
        title = "Playground — All Datasets"
        subtitle = f"{len(all_datasets)} dataset(s) loaded"
        ds_options = "".join(
            f'<option value="{d["dataset_id"]}">{d["filename"]} ({d["row_count"]} rows)</option>'
            for d in all_datasets
        )
        ds_selector = f'''<div class="ds-selector">
            <label>Dataset:</label>
            <select id="dsInput" multiple>{ds_options}</select>
            <small>Hold Ctrl/Cmd to select multiple datasets</small>
        </div>'''
        ds_info = ""

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1117; color: #e0e0e0; min-height: 100vh; padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 1.5rem; color: #fff; margin-bottom: 4px; }}
        .subtitle {{ color: #666; font-size: 0.85rem; margin-bottom: 20px; }}
        .ds-selector {{
            margin-bottom: 16px; padding: 12px; background: #1a1b26;
            border-radius: 8px; display: flex; align-items: center; gap: 12px;
        }}
        .ds-selector label {{ color: #888; font-weight: 600; }}
        .ds-selector select {{
            flex: 1; padding: 8px; background: #0f1117; border: 1px solid #2a2b3d;
            border-radius: 6px; color: #fff; font-size: 0.9rem;
        }}
        .ds-selector small {{ color: #555; font-size: 0.75rem; }}
        .columns-bar {{
            margin-bottom: 16px; font-size: 0.8rem; color: #555;
        }}
        .col-tag {{
            background: #1a1b26; padding: 3px 8px; border-radius: 4px;
            margin: 2px; display: inline-block; color: #7aa2f7;
        }}
        .input-row {{ display: flex; gap: 12px; margin-bottom: 16px; }}
        input[type="text"] {{
            flex: 1; padding: 14px 18px; background: #1a1b26; border: 1px solid #2a2b3d;
            border-radius: 10px; color: #fff; font-size: 1rem; outline: none;
        }}
        input[type="text"]:focus {{ border-color: #7aa2f7; }}
        button {{
            padding: 14px 28px; background: #7aa2f7; color: #0f1117; border: none;
            border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer;
        }}
        button:hover {{ background: #5d8bee; }}
        button:disabled {{ background: #333; color: #666; cursor: not-allowed; }}
        #chart {{
            background: #1a1b26; border-radius: 12px; padding: 20px;
            min-height: 400px; box-shadow: 0 4px 24px rgba(0,0,0,0.3);
            display: flex; align-items: center; justify-content: center;
        }}
        #chart.has-chart {{ display: block; }}
        .placeholder {{ text-align: center; color: #444; }}
        .placeholder p {{ font-size: 1.1rem; margin-bottom: 12px; }}
        .examples span {{
            background: #1e1f2e; padding: 4px 10px; border-radius: 6px; cursor: pointer;
            display: inline-block; margin: 3px; font-size: 0.85rem;
        }}
        .examples span:hover {{ background: #2a2b3d; color: #7aa2f7; }}
        .response-box {{
            margin-top: 16px; padding: 16px; background: #1a1b26; border-radius: 8px;
            font-size: 0.9rem; display: none; line-height: 1.6;
        }}
        .response-box.visible {{ display: block; }}
        .response-box .label {{ color: #7aa2f7; font-weight: 600; margin-bottom: 6px; }}
        .response-box .answer {{ color: #ccc; }}
        .response-box pre {{
            background: #0f1117; padding: 12px; border-radius: 6px; overflow-x: auto;
            font-size: 0.8rem; color: #9ece6a; margin-top: 8px;
        }}
        .intent-badge {{
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.75rem; font-weight: 600; margin-left: 8px;
        }}
        .intent-chart {{ background: #2d4f67; color: #7dcfff; }}
        .intent-analysis {{ background: #3b4261; color: #bb9af7; }}
        .intent-query {{ background: #2d3a27; color: #9ece6a; }}
        .intent-clarify {{ background: #4a3524; color: #e0af68; }}
        .clarify-btn {{
            display: inline-block; margin: 4px; padding: 6px 14px;
            background: #2a2b3d; border: 1px solid #3a3b4d; border-radius: 6px;
            color: #7aa2f7; cursor: pointer; font-size: 0.85rem;
        }}
        .clarify-btn:hover {{ background: #3a3b4d; }}
        .status {{ color: #f7768e; font-size: 0.85rem; margin-bottom: 8px; min-height: 20px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Data Playground</h1>
    <p class="subtitle">{subtitle}</p>

    {ds_selector}
    {ds_info}

    <div class="input-row">
        <input type="text" id="prompt" placeholder="Ask anything — chart, analysis, or question..."
               onkeydown="if(event.key==='Enter') send()">
        <button id="btn" onclick="send()">Send</button>
    </div>
    <div class="status" id="status"></div>

    <div id="chart">
        <div class="placeholder" id="placeholder">
            <p>Ask me anything about your data</p>
            <div class="examples">
                <span onclick="fill(this.textContent)">show me a bar chart</span>
                <span onclick="fill(this.textContent)">summary statistics</span>
                <span onclick="fill(this.textContent)">what was the highest value?</span>
                <span onclick="fill(this.textContent)">line graph of trends</span>
                <span onclick="fill(this.textContent)">compare all columns</span>
                <span onclick="fill(this.textContent)">pie chart breakdown</span>
                <span onclick="fill(this.textContent)">top 5 rows by revenue</span>
                <span onclick="fill(this.textContent)">correlation heatmap</span>
            </div>
        </div>
    </div>

    <div class="response-box" id="response"></div>
</div>

<script>
    function fill(text) {{
        document.getElementById('prompt').value = text;
        send();
    }}

    function getDatasetIds() {{
        const el = document.getElementById('dsInput');
        if (el.tagName === 'INPUT') return [el.value];
        return Array.from(el.selectedOptions).map(o => o.value);
    }}

    async function send() {{
        const prompt = document.getElementById('prompt').value.trim();
        if (!prompt) return;

        const btn = document.getElementById('btn');
        const status = document.getElementById('status');
        const responseBox = document.getElementById('response');

        btn.disabled = true;
        btn.textContent = 'Thinking...';
        status.textContent = '';

        try {{
            const body = {{ message: prompt }};
            const dsIds = getDatasetIds();
            if (dsIds.length > 0) body.dataset_ids = dsIds;

            const resp = await fetch('/api/chat', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(body)
            }});

            if (!resp.ok) {{
                const err = await resp.json();
                throw new Error(err.detail || 'Request failed');
            }}

            const data = await resp.json();
            handleResponse(data);

        }} catch (err) {{
            status.textContent = err.message;
        }} finally {{
            btn.disabled = false;
            btn.textContent = 'Send';
        }}
    }}

    function handleResponse(data) {{
        const responseBox = document.getElementById('response');
        const chartDiv = document.getElementById('chart');
        const status = document.getElementById('status');

        const badgeClass = 'intent-' + data.intent;
        let html = `<div class="label">${{data.intent.toUpperCase()}} <span class="intent-badge ${{badgeClass}}">${{data.intent}}</span></div>`;
        html += `<div class="answer">${{data.message}}</div>`;

        // Handle chart
        if (data.intent === 'chart' && data.chart) {{
            const ph = document.getElementById('placeholder');
            if (ph) ph.remove();
            chartDiv.classList.add('has-chart');

            const layout = data.chart.layout || {{}};
            layout.paper_bgcolor = '#1a1b26';
            layout.plot_bgcolor = '#1a1b26';
            layout.font = {{ color: '#e0e0e0', family: '-apple-system, sans-serif' }};
            if (layout.xaxis) {{ layout.xaxis.gridcolor = '#2a2b3d'; layout.xaxis.color = '#888'; }}
            if (layout.yaxis) {{ layout.yaxis.gridcolor = '#2a2b3d'; layout.yaxis.color = '#888'; }}

            Plotly.newPlot('chart', data.chart.data, layout, {{
                responsive: true, displayModeBar: true, displaylogo: false
            }});

            status.style.color = '#9ece6a';
            status.textContent = 'Chart generated';
        }}

        // Handle analysis
        if (data.intent === 'analysis' && data.analysis) {{
            html += `<pre>${{JSON.stringify(data.analysis, null, 2)}}</pre>`;
        }}

        // Handle query
        if (data.intent === 'query' && data.data) {{
            html += `<pre>${{JSON.stringify(data.data, null, 2)}}</pre>`;
        }}

        // Handle clarification
        if (data.intent === 'clarify' && data.options) {{
            html += '<div style="margin-top:10px">';
            data.options.forEach(opt => {{
                html += `<span class="clarify-btn" onclick="selectDataset('${{opt.dataset_id}}', '${{opt.filename}}')">${{opt.filename}}</span>`;
            }});
            html += '</div>';
            status.style.color = '#e0af68';
            status.textContent = 'Please select a dataset';
        }}

        responseBox.innerHTML = html;
        responseBox.classList.add('visible');
    }}

    function selectDataset(id, name) {{
        const el = document.getElementById('dsInput');
        if (el.tagName === 'SELECT') {{
            Array.from(el.options).forEach(o => o.selected = o.value === id);
        }}
        const prompt = document.getElementById('prompt').value;
        if (prompt) send();
    }}
</script>
</body>
</html>"""


# ─── Run ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)