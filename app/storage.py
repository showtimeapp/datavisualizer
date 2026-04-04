"""
Storage — permanent (parquet) + temporary (in-memory) datasets.
Temp datasets auto-expire after TTL and are never saved to disk.
"""

import json
import uuid
import time
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

STORE_DIR = Path(settings.DATASETS_DIR)
STORE_DIR.mkdir(exist_ok=True)

# ─── Temp Storage (in-memory, auto-expiring) ─────────────

TEMP_STORE: dict[str, dict] = {}      # {dataset_id: {df, meta, created_at, last_accessed, ttl}}
TEMP_TTL_SECONDS = 30 * 60            # 30 minutes of inactivity → auto-delete
CLEANUP_INTERVAL = 60                  # Check for expired sessions every 60s
_cleanup_started = False


def _start_cleanup_thread():
    """Background thread that removes expired temp datasets."""
    global _cleanup_started
    if _cleanup_started:
        return
    _cleanup_started = True

    def cleanup_loop():
        while True:
            time.sleep(CLEANUP_INTERVAL)
            now = time.time()
            expired = [
                did for did, entry in TEMP_STORE.items()
                if now - entry["last_accessed"] > entry["ttl"]
            ]
            for did in expired:
                del TEMP_STORE[did]
                logger.info(f"Temp dataset '{did}' expired and removed")

    t = threading.Thread(target=cleanup_loop, daemon=True)
    t.start()


_start_cleanup_thread()


# ─── ID Generation ───────────────────────────────────────

def generate_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


# ═══════════════════════════════════════════════════════════
#  PERMANENT STORAGE (parquet on disk)
# ═══════════════════════════════════════════════════════════

def save_dataset(df: pd.DataFrame, filename: str, dataset_id: Optional[str] = None) -> str:
    """Save a DataFrame permanently. Returns dataset_id."""
    did = dataset_id or generate_id()
    folder = STORE_DIR / did
    folder.mkdir(exist_ok=True)

    df.to_parquet(folder / "data.parquet", index=False)

    columns = _build_column_meta(df)
    meta = {
        "dataset_id": did,
        "filename": filename,
        "columns": columns,
        "row_count": len(df),
        "temporary": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(folder / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return did


# ═══════════════════════════════════════════════════════════
#  TEMPORARY STORAGE (in-memory, auto-expires)
# ═══════════════════════════════════════════════════════════

def save_temp_dataset(df: pd.DataFrame, name: str, ttl: int = TEMP_TTL_SECONDS) -> str:
    """
    Save a DataFrame in memory only. Auto-deletes after `ttl` seconds of inactivity.
    Returns dataset_id prefixed with 'tmp_'.
    """
    did = generate_id(prefix="tmp_")
    columns = _build_column_meta(df)

    TEMP_STORE[did] = {
        "df": df,
        "meta": {
            "dataset_id": did,
            "filename": name,
            "columns": columns,
            "row_count": len(df),
            "temporary": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "created_at": time.time(),
        "last_accessed": time.time(),
        "ttl": ttl,
    }

    logger.info(f"Temp dataset '{did}' created ({len(df)} rows, TTL={ttl}s)")
    return did


def touch_temp(dataset_id: str):
    """Reset the expiry timer for a temp dataset (called on each access)."""
    if dataset_id in TEMP_STORE:
        TEMP_STORE[dataset_id]["last_accessed"] = time.time()


# ═══════════════════════════════════════════════════════════
#  UNIFIED ACCESS (checks temp first, then permanent)
# ═══════════════════════════════════════════════════════════

def load_dataset(dataset_id: str) -> pd.DataFrame:
    """Load DataFrame — checks temp memory first, then disk."""
    # Check temp store
    if dataset_id in TEMP_STORE:
        touch_temp(dataset_id)
        return TEMP_STORE[dataset_id]["df"].copy()

    # Check permanent store
    path = STORE_DIR / dataset_id / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
    return pd.read_parquet(path)


def load_metadata(dataset_id: str) -> dict:
    """Load metadata — checks temp memory first, then disk."""
    if dataset_id in TEMP_STORE:
        touch_temp(dataset_id)
        return TEMP_STORE[dataset_id]["meta"].copy()

    path = STORE_DIR / dataset_id / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
    with open(path) as f:
        return json.load(f)


def list_datasets(include_temp: bool = True) -> list[dict]:
    """List all datasets — both permanent and temp."""
    results = []

    # Permanent datasets
    if STORE_DIR.exists():
        for folder in STORE_DIR.iterdir():
            meta_path = folder / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                results.append({
                    "dataset_id": meta["dataset_id"],
                    "filename": meta["filename"],
                    "row_count": meta["row_count"],
                    "temporary": False,
                    "created_at": meta["created_at"],
                })

    # Temp datasets
    if include_temp:
        for did, entry in TEMP_STORE.items():
            remaining = max(0, int(entry["ttl"] - (time.time() - entry["last_accessed"])))
            results.append({
                "dataset_id": did,
                "filename": entry["meta"]["filename"],
                "row_count": entry["meta"]["row_count"],
                "temporary": True,
                "expires_in_seconds": remaining,
                "created_at": entry["meta"]["created_at"],
            })

    return results


def delete_dataset(dataset_id: str) -> bool:
    """Delete a dataset (temp or permanent)."""
    # Check temp
    if dataset_id in TEMP_STORE:
        del TEMP_STORE[dataset_id]
        logger.info(f"Temp dataset '{dataset_id}' deleted")
        return True

    # Check permanent
    folder = STORE_DIR / dataset_id
    if not folder.exists():
        return False
    import shutil
    shutil.rmtree(folder)
    return True


# ─── Helpers ─────────────────────────────────────────────

def _build_column_meta(df: pd.DataFrame) -> list[dict]:
    """Build column metadata list."""
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
    return columns