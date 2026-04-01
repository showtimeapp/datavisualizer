"""
Processing — cleans financial data, detects types, normalizes values.
Runs after ingestion, before storage.
"""

import re
import pandas as pd
import numpy as np


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: clean → detect types → normalize."""
    df = df.copy()
    df = _strip_whitespace(df)
    df = _clean_financial_values(df)
    df = _auto_convert_types(df)
    df = _handle_missing_values(df)
    return df


# ─── Cleaning Steps ─────────────────────────────────────

def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string values."""
    df.columns = [col.strip() for col in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _clean_financial_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert $1,200.50 → 1200.50, 15% → 15, (500) → -500, etc."""
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(20).astype(str)

        # Check if column looks numeric with financial formatting
        financial_pattern = re.compile(
            r'^[\s$€£¥₹]*[\(\-]?[\d,]+\.?\d*[%MmBbKkTt]?\)?\s*$'
        )
        match_count = sample.apply(lambda x: bool(financial_pattern.match(str(x)))).sum()

        if match_count < len(sample) * 0.6:
            continue  # Not a financial column

        df[col] = df[col].apply(_parse_financial_value)
    return df


def _parse_financial_value(val) -> object:
    """Parse a single financial string into a number."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()

    if s.lower() in ("nan", "none", "null", "", "n/a", "-"):
        return np.nan

    # Detect negative: (500) or -500
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    if s.startswith("-"):
        negative = True
        s = s[1:]

    # Remove currency symbols
    s = re.sub(r'[$€£¥₹]', '', s).strip()

    # Handle suffixes: K=1e3, M=1e6, B=1e9, T=1e12
    multiplier = 1
    suffix_map = {'k': 1e3, 'm': 1e6, 'b': 1e9, 't': 1e12}
    if s and s[-1].lower() in suffix_map:
        multiplier = suffix_map[s[-1].lower()]
        s = s[:-1]

    # Handle percentage
    is_percent = s.endswith('%')
    if is_percent:
        s = s[:-1]

    # Remove commas
    s = s.replace(',', '')

    try:
        num = float(s) * multiplier
        if negative:
            num = -num
        return num
    except ValueError:
        return val  # Return original if unparseable


def _auto_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Try to convert object columns to numeric or datetime."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Try numeric
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > len(df) * 0.5:
            df[col] = converted
            continue

        # Try datetime
        try:
            # Try common date formats first, then fall back
            sample = df[col].dropna().head(5).astype(str)
            dt = None
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y"]:
                try:
                    test = pd.to_datetime(sample, format=fmt, errors="raise")
                    dt = pd.to_datetime(df[col], format=fmt, errors="coerce")
                    break
                except (ValueError, TypeError):
                    continue
            if dt is None:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().sum() > len(df) * 0.5:
                df[col] = dt
                continue
        except Exception:
            pass

    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common missing markers with NaN."""
    missing_markers = ["", "N/A", "n/a", "NA", "null", "None", "-", "--", "nan"]
    df = df.replace(missing_markers, np.nan)
    return df