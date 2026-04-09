# kalshi.py

import pandas as pd

from config import KALSHI_BASE
from helpers import safe_get_json, to_dt, parse_mixed_utc


def fetch_kalshi_historical_markets(limit=100, cursor=None, exclude_sports=False):
    params = {"limit": limit}
    if cursor:
        params["cursor"] = cursor

    data = safe_get_json(f"{KALSHI_BASE}/historical/markets", params=params)

    markets = data.get("markets", [])
    rows = []

    for m in markets:
        rows.append({
            "ticker": m.get("ticker"),
            "title": m.get("title"),
            "subtitle": m.get("subtitle"),
            "status": m.get("status"),
            "result": m.get("result"),
            "close_time": m.get("close_time"),
            "expiration_time": m.get("expiration_time"),
        })

    return pd.DataFrame(rows), data.get("cursor")


def fetch_kalshi_candles(ticker, start_date, end_date):
    params = {"period_interval": 1440}

    data = safe_get_json(
        f"{KALSHI_BASE}/historical/markets/{ticker}/candlesticks",
        params=params
    )

    candles = data.get("candlesticks", [])
    rows = []

    for c in candles:
        rows.append({
            "ticker": ticker,
            "timestamp_utc": to_dt(c.get("end_period_ts") or c.get("ts")),
            "price_yes": c.get("close"),
            "open": c.get("open"),
            "high": c.get("high"),
            "low": c.get("low"),
            "volume": c.get("volume"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp_utc"] = parse_mixed_utc(df["timestamp_utc"])
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)

    df = df[df["timestamp_utc"].notna()].copy()
    df = df[(df["timestamp_utc"] >= start_ts) & (df["timestamp_utc"] <= end_ts)]
    return df.sort_values("timestamp_utc").reset_index(drop=True)


if __name__ == "__main__":
    df, _ = fetch_kalshi_historical_markets(limit=20)
    print(df.head(20).to_string(index=False))