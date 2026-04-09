import pandas as pd

from config import KALSHI_BASE, MAX_KALSHI_MARKETS
from utils import save_csv, safe_get


def fetch_kalshi_markets(limit: int = MAX_KALSHI_MARKETS) -> pd.DataFrame:
    print("Fetching Kalshi markets...")
    rows = []
    cursor = None
    fetched = 0

    while fetched < limit:
        params = {"limit": min(1000, limit - fetched)}
        if cursor:
            params["cursor"] = cursor

        payload = safe_get(f"{KALSHI_BASE}/markets", params=params)
        markets = payload.get("markets", [])
        cursor = payload.get("cursor")

        for m in markets:
            rows.append({
                "market_id": m.get("ticker"),
                "title": m.get("title"),
                "event_ticker": m.get("event_ticker"),
                "series_ticker": m.get("series_ticker"),
                "status": m.get("status"),
                "close_time": m.get("close_time"),
                "expiration_time": m.get("expiration_time"),
                "yes_ask": m.get("yes_ask"),
                "yes_bid": m.get("yes_bid"),
                "last_price": m.get("last_price"),
                "volume": m.get("volume"),
            })

        fetched += len(markets)
        if not cursor or not markets:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        save_csv(df, "data/processed/kalshi_markets.csv")
    else:
        print("Warning: Kalshi markets metadata empty.")
    return df


def _fetch_live_candles(ticker: str) -> list[dict]:
    url = f"{KALSHI_BASE}/series/{ticker.split('-')[0]}/markets/{ticker}/candlesticks"
    try:
        payload = safe_get(url, params={"period_interval": 1440})
        return payload.get("candlesticks", [])
    except Exception:
        return []


def _fetch_historical_candles(ticker: str) -> list[dict]:
    url = f"{KALSHI_BASE}/historical/markets/{ticker}/candlesticks"
    try:
        payload = safe_get(url, params={"period_interval": 1440})
        return payload.get("candlesticks", [])
    except Exception:
        return []


def fetch_kalshi_daily_candles(markets_df: pd.DataFrame) -> None:
    print("Fetching Kalshi daily candlesticks...")
    rows = []

    for _, row in markets_df.iterrows():
        ticker = row.get("market_id")
        if not ticker:
            continue

        candles = _fetch_live_candles(ticker)
        if not candles:
            candles = _fetch_historical_candles(ticker)

        for c in candles:
            ts = c.get("end_period_ts") or c.get("start_period_ts") or c.get("ts")
            close_price = c.get("close") or c.get("close_price") or c.get("price")
            if ts is None or close_price is None:
                continue

            # Kalshi timestamps may be ms or s
            unit = "ms" if len(str(int(ts))) >= 13 else "s"

            rows.append({
                "market_id": ticker,
                "Date": pd.to_datetime(ts, unit=unit).floor("D"),
                "kalshi_prob": float(close_price) / 100.0 if float(close_price) > 1 else float(close_price),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["market_id", "Date"]).drop_duplicates(["market_id", "Date"])
        save_csv(df, "data/processed/kalshi_prices_daily.csv")
    else:
        print("Warning: no Kalshi daily candles fetched.")


if __name__ == "__main__":
    mdf = fetch_kalshi_markets()
    fetch_kalshi_daily_candles(mdf)