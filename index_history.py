# index_history.py

import time
import pandas as pd
from datetime import datetime, timezone

from country_indexes import COUNTRY_INDEXES
from helpers import safe_get_json
from config import OUTPUT_INDEX_HISTORY


START_DATE = "2020-01-01"
END_DATE = pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def to_unix_seconds(date_str):
    ts = pd.to_datetime(date_str, utc=True)
    return int(ts.timestamp())


def fetch_yahoo_chart_history(symbol, start_date, end_date, interval="1d"):
    """
    Uses Yahoo chart endpoint with explicit period1/period2.
    More reliable for historical pulls than quote endpoint.
    """
    period1 = to_unix_seconds(start_date)
    # add one day so end date is included
    period2 = to_unix_seconds(
        (pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits,capitalGains",
    }

    data = safe_get_json(url, params=params)

    result = ((data or {}).get("chart") or {}).get("result") or []
    if not result:
        return pd.DataFrame()

    r = result[0]
    meta = r.get("meta") or {}
    timestamps = r.get("timestamp") or []
    quote = (((r.get("indicators") or {}).get("quote")) or [{}])[0]

    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce"),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })

    if df.empty:
        return df

    df = df[df["timestamp_utc"].notna()].copy().reset_index(drop=True)
    df["symbol"] = symbol
    df["currency"] = meta.get("currency")
    df["exchange_name"] = meta.get("exchangeName")
    df["instrument_type"] = meta.get("instrumentType")
    return df


def build_index_history(start_date=START_DATE, end_date=END_DATE, sleep_seconds=1.0):
    frames = []

    for row in COUNTRY_INDEXES:
        country = row["country"]
        index_name = row["index_name"]
        symbol = row["symbol"]
        bucket = row.get("bucket")

        print(f"Fetching {country} | {index_name} | {symbol}")

        try:
            df = fetch_yahoo_chart_history(symbol, start_date, end_date, interval="1d")
        except Exception as e:
            print(f"Failed {symbol}: {e}")
            time.sleep(sleep_seconds)
            continue

        if df.empty:
            print(f"No data for {symbol}")
            time.sleep(sleep_seconds)
            continue

        df["country"] = country
        df["index_name"] = index_name
        df["bucket"] = bucket

        frames.append(df)
        print(f"Saved in memory: {symbol} | rows={len(df)}")
        time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["country", "index_name", "timestamp_utc"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = build_index_history()

    if df.empty:
        print("No index history fetched.")
    else:
        df.to_csv(OUTPUT_INDEX_HISTORY, index=False)
        print(df.head(20).to_string(index=False))
        print(f"\nSaved: {OUTPUT_INDEX_HISTORY}")