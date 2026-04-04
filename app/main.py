"""
Financial Data Visualization Backend — Unified API v2.2
========================================================
  POST /api/upload         — Upload files (permanent)
  POST /api/text           — Send text + question → get answer (temporary, all-in-one)
  POST /api/chat           — Ask anything about uploaded data
  GET  /api/datasets       — List all datasets
  GET  /api/dataset/{id}   — Dataset details + preview
  DELETE /api/dataset/{id} — Delete a dataset
  GET  /api/playground     — Interactive chart viewer (browser)
"""

import traceback
import logging
import json
import math
import base64
from datetime import datetime, date
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response

from app.config import settings
from app.models import (
    ChatRequest, ChatResponse, UploadResponse,
    TextRequest, TextResponse,
    DatasetInfo, ColumnInfo, IntentType,
)
from app.ingestion import ingest_file, ingest_raw_text
from app.processing import process_dataframe
from app.storage import (
    save_dataset, save_temp_dataset,
    load_dataset, load_metadata,
    list_datasets, delete_dataset,
    TEMP_STORE,
)
from app.router import process_message, _unified_handler

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
    description="Upload data or send text → ask questions, get charts (PNG), run analysis.",
    version="2.2.0",
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
        "version": "2.2.0",
        "llm_provider": settings.LLM_PROVIDER,
        "endpoints": {
            "POST /api/upload": "Upload files (permanent storage)",
            "POST /api/text": "Send text + question → get answer instantly (no storage)",
            "POST /api/chat": "Ask anything about uploaded datasets",
            "GET  /api/datasets": "List all datasets",
            "GET  /api/dataset/{id}": "Dataset details + preview",
            "DELETE /api/dataset/{id}": "Delete a dataset",
            "GET  /api/playground": "Interactive chart viewer (browser)",
        },
    }


# ═══════════════════════════════════════════════════════════
#  1. UPLOAD — permanent storage, returns dataset_ids
# ═══════════════════════════════════════════════════════════

@app.post("/api/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more files. Each becomes a permanent dataset.
    Returns dataset_ids for use with /api/chat.
    
    Supported: CSV, Excel (.xlsx/.xls), HTML, Markdown, JSON, TXT
    """
    results = []
    errors = []
    dataset_ids = []
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    for file in files:
        try:
            contents = await file.read()
            if len(contents) > max_bytes:
                errors.append(f"{file.filename}: Too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)")
                continue

            df = await ingest_file(contents, file.filename)
            df = process_dataframe(df)
            dataset_id = save_dataset(df, file.filename)
            meta = load_metadata(dataset_id)

            dataset_ids.append(dataset_id)
            results.append(DatasetInfo(
                dataset_id=dataset_id,
                filename=file.filename,
                columns=[ColumnInfo(**c) for c in meta["columns"]],
                row_count=meta["row_count"],
                temporary=False,
                preview=safe_preview(df, 5),
            ))

        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {traceback.format_exc()}")
            errors.append(f"{file.filename}: {str(e)}")

    if not results and errors:
        raise HTTPException(400, detail={"message": "All uploads failed", "errors": errors})

    msg = f"Uploaded {len(results)} file(s)."
    if errors:
        msg += f" Failed: {'; '.join(errors)}"

    return UploadResponse(dataset_ids=dataset_ids, datasets=results, message=msg)


# ═══════════════════════════════════════════════════════════
#  2. TEXT — all-in-one: send data + question → get answer
# ═══════════════════════════════════════════════════════════

@app.post("/api/text", response_model=TextResponse)
async def text_endpoint(req: TextRequest):
    """
    All-in-one: send raw text data + a question.
    System parses the text, answers the question (or generates a chart PNG), 
    and returns the result. Data is NOT saved — it's processed and discarded.

    Examples:
        {
            "text": "Company,Revenue,Profit\\nApple,394,99\\nGoogle,307,73\\nMeta,134,39",
            "question": "which company has highest profit?"
        }
        {
            "text": "AAPL open 150 close 155, GOOG open 2800 close 2850, MSFT open 420 close 418",
            "question": "bar chart comparing open vs close prices"
        }
        {
            "text": "Revenue was $4.2M in Q1, $3.8M in Q2, $5.1M in Q3, $4.9M in Q4",
            "question": "line chart of revenue trend"
        }
    """
    try:
        # Step 1: Parse text into DataFrame
        df = await _smart_text_parse(req.text)
        df = process_dataframe(df)

        parsed_info = {
            "rows": len(df),
            "columns": df.columns.tolist(),
            "preview": safe_preview(df, 3),
        }

        # Step 2: Answer the question using the unified handler
        result = await _unified_handler(req.question, df, req.name, [])

        # Step 3: Build response
        return TextResponse(
            intent=result.get("intent", "query"),
            message=result.get("message", ""),
            chart_png=result.get("chart_png"),
            chart=result.get("chart"),
            chart_type=result.get("chart_type"),
            data=result.get("data"),
            parsed_data=parsed_info,
        )

    except ValueError as e:
        raise HTTPException(400, f"Could not parse text: {str(e)}")
    except Exception as e:
        logger.error(f"Text endpoint failed: {traceback.format_exc()}")
        raise HTTPException(500, f"Processing failed: {str(e)}")


async def _smart_text_parse(text: str) -> pd.DataFrame:
    """Try cheap parsing first (CSV/TSV/markdown), fall back to LLM."""
    import io
    lines = text.strip().split("\n")

    # CSV
    if len(lines) >= 2 and "," in lines[0]:
        try:
            df = pd.read_csv(io.StringIO(text))
            if len(df.columns) >= 2 and len(df) >= 1:
                return df
        except Exception:
            pass

    # TSV
    if len(lines) >= 2 and "\t" in lines[0]:
        try:
            df = pd.read_csv(io.StringIO(text), sep="\t")
            if len(df.columns) >= 2 and len(df) >= 1:
                return df
        except Exception:
            pass

    # Markdown table
    if len(lines) >= 2 and "|" in lines[0]:
        try:
            from app.ingestion import parse_markdown
            df = parse_markdown(text.encode("utf-8"))
            if len(df.columns) >= 2 and len(df) >= 1:
                return df
        except Exception:
            pass

    # LLM extraction (for sentences like "Revenue was $4.2M in Q1...")
    return await ingest_raw_text(text)


# ═══════════════════════════════════════════════════════════
#  3. CHAT — for uploaded datasets
# ═══════════════════════════════════════════════════════════

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Ask anything about uploaded datasets. System decides whether to:
    - Generate a chart (returned as PNG + Plotly JSON)
    - Answer a question
    - Ask for clarification
    """
    try:
        result = await process_message(req.message, req.dataset_ids)

        return ChatResponse(
            intent=IntentType(result.get("intent", "query")),
            message=result.get("message", ""),
            chart_png=result.get("chart_png"),
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
#  4. DATASET MANAGEMENT
# ═══════════════════════════════════════════════════════════

@app.get("/api/datasets")
async def get_datasets():
    """List all datasets."""
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
#  5. PLAYGROUND — Interactive Chart Viewer (Browser)
# ═══════════════════════════════════════════════════════════

@app.get("/api/playground/{dataset_id}", response_class=HTMLResponse)
async def playground(dataset_id: str):
    """Interactive chart playground for a specific dataset."""
    try:
        meta = load_metadata(dataset_id)
        load_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")

    columns = [c["name"] for c in meta["columns"]]
    return _build_playground_html(dataset_id, meta, columns)


@app.get("/api/playground", response_class=HTMLResponse)
async def playground_all():
    """Interactive playground for all datasets."""
    all_datasets = list_datasets()
    if not all_datasets:
        return HTMLResponse("<h1>No datasets uploaded</h1><p>Upload files via POST /api/upload or use POST /api/text</p>")
    return _build_playground_html(None, None, None, all_datasets)


def _build_playground_html(dataset_id, meta, columns, all_datasets=None):
    """Build the interactive chart playground HTML — renders PNG charts."""

    if dataset_id:
        title = f"Playground — {meta['filename']}"
        subtitle = f"{meta['filename']} — {meta['row_count']} rows"
        cols_html = " ".join(f'<span class="col-tag">{c}</span>' for c in columns)
        ds_selector = f'<input type="hidden" id="dsInput" value="{dataset_id}">'
        ds_info = f'<div class="columns-bar">Columns: {cols_html}</div>'
    else:
        title = "Playground"
        subtitle = f"{len(all_datasets)} dataset(s)"
        ds_options = "".join(
            f'<option value="{d["dataset_id"]}">{d["filename"]} ({d["row_count"]} rows)</option>'
            for d in all_datasets
        )
        ds_selector = f'''<div class="ds-selector">
            <label>Dataset:</label>
            <select id="dsInput" multiple>{ds_options}</select>
            <small>Hold Ctrl/Cmd for multiple</small>
        </div>'''
        ds_info = ""

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
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
            border-radius: 6px; color: #fff;
        }}
        .ds-selector small {{ color: #555; font-size: 0.75rem; }}
        .columns-bar {{ margin-bottom: 16px; font-size: 0.8rem; color: #555; }}
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
            min-height: 300px; box-shadow: 0 4px 24px rgba(0,0,0,0.3);
            text-align: center;
        }}
        #chart img {{
            max-width: 100%; border-radius: 8px;
        }}
        .placeholder {{ text-align: center; color: #444; padding: 80px 0; }}
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
        .response-box .answer {{ color: #ccc; white-space: pre-wrap; }}
        .response-box pre {{
            background: #0f1117; padding: 12px; border-radius: 6px; overflow-x: auto;
            font-size: 0.8rem; color: #9ece6a; margin-top: 8px; max-height: 400px;
        }}
        .intent-badge {{
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.75rem; font-weight: 600; margin-left: 8px;
        }}
        .intent-chart {{ background: #2d4f67; color: #7dcfff; }}
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
                <span onclick="fill(this.textContent)">list all columns</span>
                <span onclick="fill(this.textContent)">summary statistics</span>
                <span onclick="fill(this.textContent)">what was the highest value?</span>
                <span onclick="fill(this.textContent)">pie chart breakdown</span>
                <span onclick="fill(this.textContent)">top 5 rows</span>
            </div>
        </div>
    </div>
    <div class="response-box" id="response"></div>
</div>
<script>
    function fill(text) {{ document.getElementById('prompt').value = text; send(); }}
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
        btn.disabled = true; btn.textContent = 'Thinking...'; status.textContent = '';
        try {{
            const body = {{ message: prompt }};
            const dsIds = getDatasetIds();
            if (dsIds.length > 0) body.dataset_ids = dsIds;
            const resp = await fetch('/api/chat', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(body)
            }});
            if (!resp.ok) {{ const err = await resp.json(); throw new Error(err.detail || 'Failed'); }}
            handleResponse(await resp.json());
        }} catch (err) {{ status.textContent = err.message; }}
        finally {{ btn.disabled = false; btn.textContent = 'Send'; }}
    }}
    function handleResponse(data) {{
        const responseBox = document.getElementById('response');
        const chartDiv = document.getElementById('chart');
        const status = document.getElementById('status');
        const badgeClass = 'intent-' + data.intent;
        let html = `<div class="label">${{data.intent.toUpperCase()}} <span class="intent-badge ${{badgeClass}}">${{data.intent}}</span></div>`;
        html += `<div class="answer">${{data.message}}</div>`;

        if (data.intent === 'chart' && data.chart_png) {{
            chartDiv.innerHTML = `<img src="data:image/png;base64,${{data.chart_png}}" alt="Chart">`;
            status.style.color = '#9ece6a'; status.textContent = 'Chart generated';
        }} else if (data.intent === 'chart' && !data.chart_png) {{
            status.style.color = '#e0af68'; status.textContent = 'Chart generated (PNG unavailable)';
        }}

        if (data.intent === 'query' && data.data) {{
            html += `<pre>${{JSON.stringify(data.data, null, 2)}}</pre>`;
        }}
        if (data.intent === 'clarify' && data.options) {{
            html += '<div style="margin-top:10px">';
            data.options.forEach(opt => {{
                html += `<span class="clarify-btn" onclick="selectDataset('${{opt.dataset_id}}')">${{opt.filename}}</span>`;
            }});
            html += '</div>';
            status.style.color = '#e0af68'; status.textContent = 'Select a dataset';
        }}
        responseBox.innerHTML = html;
        responseBox.classList.add('visible');
    }}
    function selectDataset(id) {{
        const el = document.getElementById('dsInput');
        if (el.tagName === 'SELECT') Array.from(el.options).forEach(o => o.selected = o.value === id);
        if (document.getElementById('prompt').value) send();
    }}
</script>
</body>
</html>"""


# ─── Run ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    