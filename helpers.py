# helpers.py

import requests
import pandas as pd
from datetime import datetime, timezone

from config import USER_AGENT


def safe_get_json(url, params=None, headers=None, timeout=30):
    headers = headers or {}
    headers.setdefault("User-Agent", USER_AGENT)
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def safe_get_text(url, params=None, headers=None, timeout=30):
    headers = headers or {}
    headers.setdefault("User-Agent", USER_AGENT)
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def to_dt(ts):
    if ts is None:
        return None

    if isinstance(ts, (int, float)):
        if ts > 10**12:
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            try:
                return pd.to_datetime(ts, utc=True, errors="coerce")
            except Exception:
                return None

    return None


def parse_mixed_utc(series):
    return pd.to_datetime(
        series.astype(str).str.strip().replace({"nan": None, "NaT": None}),
        utc=True,
        errors="coerce",
        format="mixed"
    )