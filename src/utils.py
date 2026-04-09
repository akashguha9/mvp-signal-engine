import json
import os
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests

from config import REQUEST_TIMEOUT, SLEEP_SECONDS


def ensure_dirs() -> None:
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path} | Rows: {len(df)}")


def safe_get(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    time.sleep(SLEEP_SECONDS)
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def to_unix_ts(date_str: str) -> int:
    return int(pd.Timestamp(date_str).timestamp())


def normalize_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("D")
    return df


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def print_stage(name: str, rows: Optional[int] = None) -> None:
    if rows is None:
        print(f"[STAGE] {name}")
    else:
        print(f"[STAGE] {name} | Rows: {rows}")