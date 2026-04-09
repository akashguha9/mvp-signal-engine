# google_trends_mapper.py

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None


INPUT_FILE = "output_timeseries_dataset_dedup.csv"
OUTPUT_FILE = "output_google_trends.csv"


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "for", "on", "in", "at", "by",
    "with", "from", "is", "are", "was", "were", "be", "been", "being",
    "after", "before", "into", "over", "under", "up", "down",
    "what", "why", "how", "when", "where", "who",
    "market", "markets", "index", "indices", "stock", "stocks", "price", "prices",
}


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Missing file: {path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Empty file: {path}")
        return pd.DataFrame()

    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce", format="mixed")

    return df


def extract_query(row: pd.Series) -> str:
    candidates = []

    for col in ["seed_label", "symbol", "reason"]:
        if col in row.index and pd.notna(row.get(col)):
            candidates.append(str(row.get(col)))

    raw = " ".join(candidates).lower()
    raw = re.sub(r"[^a-zA-Z0-9_\-\s]", " ", raw)
    toks = [t for t in raw.split() if t not in STOPWORDS and len(t) > 2]

    # prioritize a symbol if present
    symbol = str(row.get("symbol", "")).strip()
    if symbol and symbol != "nan":
        symbol_clean = re.sub(r"[^A-Za-z0-9]", "", symbol)
        if symbol_clean:
            toks = [symbol_clean] + toks

    toks = list(dict.fromkeys(toks))
    return " ".join(toks[:5]).strip()


def fetch_trends(pytrends: TrendReq, query: str, timeframe: str = "today 3-m") -> dict:
    try:
        pytrends.build_payload([query], cat=0, timeframe=timeframe, geo="", gprop="")
        iot = pytrends.interest_over_time()

        if iot is None or iot.empty or query not in iot.columns:
            return {
                "trend_query": query,
                "trend_mean": None,
                "trend_last": None,
                "trend_max": None,
                "trend_min": None,
                "trend_status": "empty",
            }

        series = iot[query].dropna()
        if series.empty:
            return {
                "trend_query": query,
                "trend_mean": None,
                "trend_last": None,
                "trend_max": None,
                "trend_min": None,
                "trend_status": "empty_series",
            }

        return {
            "trend_query": query,
            "trend_mean": float(series.mean()),
            "trend_last": float(series.iloc[-1]),
            "trend_max": float(series.max()),
            "trend_min": float(series.min()),
            "trend_status": "ok",
        }
    except Exception as e:
        return {
            "trend_query": query,
            "trend_mean": None,
            "trend_last": None,
            "trend_max": None,
            "trend_min": None,
            "trend_status": f"error: {type(e).__name__}",
        }


def main():
    print("=== GOOGLE TRENDS MAPPER ===")
    print(f"input_file: {INPUT_FILE}")

    df = load_data(INPUT_FILE)
    if df.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False)
        print("No data.")
        return

    if TrendReq is None:
        print("pytrends is not installed. Run: pip install pytrends")
        return

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.3)

    out_rows = []
    cache: dict[str, dict] = {}

    # limit initial research pass
    sample = df.copy()
    if len(sample) > 150:
        sample = sample.head(150)

    for _, row in sample.iterrows():
        query = extract_query(row)

        if not query:
            result = {
                "trend_query": None,
                "trend_mean": None,
                "trend_last": None,
                "trend_max": None,
                "trend_min": None,
                "trend_status": "no_query",
            }
        else:
            if query in cache:
                result = cache[query]
            else:
                result = fetch_trends(pytrends, query=query, timeframe="today 3-m")
                cache[query] = result
                time.sleep(1.2)

        out_rows.append({
            "event_time": row.get("event_time"),
            "seed_label": row.get("seed_label"),
            "symbol": row.get("symbol"),
            "reason": row.get("reason"),
            **result,
        })

    out = pd.DataFrame(out_rows)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"rows written: {len(out)}")
    if "trend_status" in out.columns:
        print("\nTrend status counts:")
        print(out["trend_status"].value_counts(dropna=False).to_string())

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()