import pandas as pd

from config import KALSHI_BASE, MAX_KALSHI_MARKETS
from utils import save_csv, safe_get, safe_float, print_stage


def fetch_kalshi_markets(limit: int = MAX_KALSHI_MARKETS) -> pd.DataFrame:
    print_stage("Fetching Kalshi markets")
    rows = []
    cursor = None
    fetched = 0

    while fetched < limit:
        params = {"limit": min(1000, limit - fetched)}
        if cursor:
            params["cursor"] = cursor

        try:
            payload = safe_get(f"{KALSHI_BASE}/markets", params=params)
        except Exception as e:
            print(f"Kalshi market fetch failed: {e}")
            break

        markets = payload.get("markets", []) if isinstance(payload, dict) else []
        cursor = payload.get("cursor") if isinstance(payload, dict) else None

        for m in markets:
            rows.append({
                "market_id": m.get("ticker"),
                "title": m.get("title"),
                "subtitle": m.get("subtitle"),
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
        df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
        df["expiration_time"] = pd.to_datetime(df["expiration_time"], errors="coerce")
    save_csv(df, "data/processed/kalshi_markets.csv")
    print_stage("Kalshi markets complete", len(df))
    return df


def _parse_candles(payload) -> list[dict]:
    if isinstance(payload, dict):
        candles = payload.get("candlesticks", [])
        if isinstance(candles, list):
            return candles
    return []


def _fetch_live_candles(series_ticker: str, ticker: str) -> list[dict]:
    if not series_ticker or not ticker:
        return []
    url = f"{KALSHI_BASE}/series/{series_ticker}/markets/{ticker}/candlesticks"
    try:
        payload = safe_get(url, params={"period_interval": 1440})
        return _parse_candles(payload)
    except Exception as e:
        print(f"Kalshi live candles failed for {ticker}: {e}")
        return []


def _fetch_historical_candles(ticker: str) -> list[dict]:
    if not ticker:
        return []
    url = f"{KALSHI_BASE}/historical/markets/{ticker}/candlesticks"
    try:
        payload = safe_get(url, params={"period_interval": 1440})
        return _parse_candles(payload)
    except Exception as e:
        print(f"Kalshi historical candles failed for {ticker}: {e}")
        return []


def _parse_kalshi_ts(ts) -> pd.Timestamp | None:
    if ts is None:
        return None
    try:
        unit = "ms" if len(str(int(ts))) >= 13 else "s"
        return pd.to_datetime(ts, unit=unit).floor("D")
    except Exception:
        return None


def _parse_kalshi_close(candle: dict) -> float | None:
    close_price = (
        candle.get("close")
        or candle.get("close_price")
        or candle.get("price")
        or candle.get("yes_price")
    )
    val = safe_float(close_price)
    if val is None:
        return None
    if val > 1:
        val = val / 100.0
    return val


def fetch_kalshi_daily_candles(markets_df: pd.DataFrame) -> None:
    print_stage("Fetching Kalshi daily candlesticks")

    if markets_df.empty:
        save_csv(pd.DataFrame(), "data/processed/kalshi_prices_daily.csv")
        print("Warning: no Kalshi markets available for daily candles.")
        return

    rows = []
    attempted = 0
    successful = 0

    for _, row in markets_df.iterrows():
        ticker = row.get("market_id")
        series_ticker = row.get("series_ticker")

        if not ticker:
            continue

        attempted += 1
        candles = _fetch_live_candles(series_ticker, ticker)
        if not candles:
            candles = _fetch_historical_candles(ticker)

        market_had_rows = False
        for c in candles:
            ts = c.get("end_period_ts") or c.get("start_period_ts") or c.get("ts") or c.get("timestamp")
            dt = _parse_kalshi_ts(ts)
            prob = _parse_kalshi_close(c)

            if dt is None or prob is None:
                continue

            rows.append({
                "market_id": ticker,
                "series_ticker": series_ticker,
                "Date": dt,
                "kalshi_prob": prob,
            })
            market_had_rows = True

        if market_had_rows:
            successful += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["market_id", "Date"]).drop_duplicates(["market_id", "Date"])
    save_csv(df, "data/processed/kalshi_prices_daily.csv")

    print_stage("Kalshi daily candles complete", len(df))
    print(f"Kalshi attempted markets: {attempted}")
    print(f"Kalshi successful markets: {successful}")