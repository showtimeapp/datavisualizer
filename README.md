# Financial Data Visualization API — v2

One smart endpoint. Upload data, ask anything — the LLM decides whether to chart, analyze, or answer.

## Quick Start

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # Set your API key
uvicorn app.main:app --reload --port 8000
```

## API (just 3 core endpoints)

### 1. Upload Files
```
POST /api/upload
```
Upload one or more files at once (CSV, Excel, HTML, Markdown, JSON, TXT):
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "files=@sales.csv" \
  -F "files=@expenses.xlsx"
```

### 2. Chat (the main endpoint)
```
POST /api/chat
```
Ask anything — LLM automatically routes to chart/analysis/query:
```bash
# Chart
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "bar chart of revenue by quarter"}'

# Analysis
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "give me summary statistics"}'

# Question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what month had the highest revenue?"}'

# Specify datasets
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "compare revenue", "dataset_ids": ["abc123", "def456"]}'
```

### 3. Playground (browser)
```
GET /api/playground/{dataset_id}   — for one dataset
GET /api/playground                — for all datasets
```

### Dataset Management
```
GET    /api/datasets         — List all
GET    /api/dataset/{id}     — Details + preview
DELETE /api/dataset/{id}     — Remove
```

## Response Format

Every `/api/chat` response has the same shape:
```json
{
  "intent": "chart | analysis | query | clarify",
  "message": "Human-readable explanation",
  "chart": { Plotly JSON spec or null },
  "chart_type": "bar",
  "chart_config": { "x": "...", "y": [...] },
  "analysis": { stats data or null },
  "data": { query result or null },
  "options": [ list of datasets if clarification needed ],
  "datasets_used": ["id1"]
}
```

## Using Plotly JSON in Your Frontend

The `chart` field contains a full Plotly spec. Render it:

**React:**
```jsx
import Plot from 'react-plotly.js';
<Plot data={response.chart.data} layout={response.chart.layout} />
```

**Vanilla JS:**
```html
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<div id="chart"></div>
<script>
  Plotly.newPlot('chart', response.chart.data, response.chart.layout);
</script>
```

**Next.js:**
```jsx
import dynamic from 'next/dynamic';
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });
```

## Project Structure
```
app/
├── main.py          # FastAPI app + routes
├── router.py        # LLM brain — classifies intent, resolves datasets
├── config.py        # Settings
├── models.py        # Request/response schemas
├── ingestion.py     # File parsers
├── processing.py    # Financial data cleaning
├── analysis.py      # Stats engine
├── charts.py        # Chart builders + smart chart
├── llm_client.py    # OpenAI/Gemini client with retry
└── storage.py       # Dataset persistence
```