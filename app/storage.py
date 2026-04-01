"""
Simple file-based storage for datasets.
Stores DataFrames as parquet and metadata as JSON.
"""

import json
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import settings

STORE_DIR = Path(settings.DATASETS_DIR)
STORE_DIR.mkdir(exist_ok=True)


def generate_id() -> str:
    return uuid.uuid4().hex[:10]


def save_dataset(df: pd.DataFrame, filename: str, dataset_id: Optional[str] = None) -> str:
    """Save a DataFrame + metadata. Returns dataset_id."""
    did = dataset_id or generate_id()
    folder = STORE_DIR / did
    folder.mkdir(exist_ok=True)

    # Save data
    df.to_parquet(folder / "data.parquet", index=False)

    # Build column metadata
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        role = "measure" if is_numeric else "dimension"
        samples = df[col].dropna().head(3).astype(str).tolist()
        columns.append({
            "name": col,
            "dtype": dtype,
            "role": role,
            "sample_values": samples,
        })

    meta = {
        "dataset_id": did,
        "filename": filename,
        "columns": columns,
        "row_count": len(df),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(folder / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return did


def load_dataset(dataset_id: str) -> pd.DataFrame:
    """Load DataFrame by dataset_id."""
    path = STORE_DIR / dataset_id / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
    return pd.read_parquet(path)


def load_metadata(dataset_id: str) -> dict:
    """Load metadata by dataset_id."""
    path = STORE_DIR / dataset_id / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
    with open(path) as f:
        return json.load(f)


def list_datasets() -> list[dict]:
    """List all stored datasets (id + filename + row_count)."""
    results = []
    if not STORE_DIR.exists():
        return results
    for folder in STORE_DIR.iterdir():
        meta_path = folder / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            results.append({
                "dataset_id": meta["dataset_id"],
                "filename": meta["filename"],
                "row_count": meta["row_count"],
                "created_at": meta["created_at"],
            })
    return results


def delete_dataset(dataset_id: str) -> bool:
    """Delete a dataset."""
    folder = STORE_DIR / dataset_id
    if not folder.exists():
        return False
    import shutil
    shutil.rmtree(folder)
    return True
