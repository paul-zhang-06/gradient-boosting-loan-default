"""Load the Home Credit Default Risk tables with memory-friendly dtypes and the main data-quality fixes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
SENTINEL_DAYS = 365243          # Home Credit uses 365243 for "not applicable" in DAYS_* columns


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, int64 -> smallest int; keeps tables of 10M+ rows within RAM."""
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[c]) and c not in ("SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def read(name: str, raw: Path = RAW, **kw) -> pd.DataFrame:
    return downcast(pd.read_csv(raw / f"{name}.csv", **kw))


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known anomalies: DAYS_EMPLOYED = 365243 (18% of rows, pensioners/unemployed) -> NaN + flag; CODE_GENDER 'XNA' -> NaN."""
    df = df.copy()
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == SENTINEL_DAYS).astype(np.int8)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(SENTINEL_DAYS, np.nan)
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
    return df


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
