import pandas as pd

from config import KALSHI_BASE, MAX_KALSHI_MARKETS
from utils import save_csv, safe_get, safe_float, print_stage


RELEVANT_TERMS = {
    "inflation", "fed", "fomc", "cpi", "jobs", "payrolls", "recession", "gdp",
    "election", "president", "senate", "house", "war", "ukraine", "russia",
    "china", "tariff", "sanction", "sanctions", "oil", "treasury", "rate", "rates",
    "default", "debt", "ceasefire", "iran", "israel", "bitcoin", "btc", "ethereum", "eth"
}

IRRELEVANT_TERMS = {
    "nba", "nfl", "mlb", "nhl", "soccer", "football", "baseball", "basketball",
    "tennis", "golf", "march madness", "championship", "game", "match", "player",
    "album", "movie", "oscar", "grammy", "celebrity", "weather"
}


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _is_relevant_market(title: str, subtitle: str, event_ticker: str, series_ticker: str) -> bool:
    text = " ".join([_norm(title), _norm(subtitle), _norm(event_ticker), _norm(series_ticker)])

    if any(term in text for term in IRRELEVANT_TERMS):
        return False

    return any(term in text for term in RELEVANT_TERMS)


def _parse_candles(payload) -> list[dict]:
    if isinstance(payload, dict):
        for key in ["candlesticks", "data", "candles"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
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
        or candle.get("end_price")
    )
    val = safe_float(close_price)
    if val is None:
        return None
    if val > 1:
        val = val / 100.0
    return val


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

        if not markets:
            break

        for m in markets:
            title = m.get("title")
            subtitle = m.get("subtitle")
            event_ticker = m.get("event_ticker")
            series_ticker = m.get("series_ticker")

            rows.append({
                "market_id": m.get("ticker"),
                "title": title,
                "subtitle": subtitle,
                "event_ticker": event_ticker,
                "series_ticker": series_ticker,
                "status": m.get("status"),
                "close_time": m.get("close_time"),
                "expiration_time": m.get("expiration_time"),
                "yes_ask": m.get("yes_ask"),
                "yes_bid": m.get("yes_bid"),
                "last_price": m.get("last_price"),
                "volume": m.get("volume"),
                "is_relevant": _is_relevant_market(title, subtitle, event_ticker, series_ticker),
            })

        fetched += len(markets)
        if not cursor:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce")
        df["expiration_time"] = pd.to_datetime(df["expiration_time"], errors="coerce")

    save_csv(df, "data/processed/kalshi_markets.csv")

    total = len(df)
    relevant = int(df["is_relevant"].fillna(False).sum()) if not df.empty else 0
    print(f"Kalshi total markets fetched: {total}")
    print(f"Kalshi relevant markets kept: {relevant}")

    return df


def _fetch_live_candles(series_ticker: str, ticker: str) -> tuple[list[dict], str]:
    if not series_ticker or not ticker:
        return [], ""
    url = f"{KALSHI_BASE}/series/{series_ticker}/markets/{ticker}/candlesticks"
    try:
        payload = safe_get(url, params={"period_interval": 1440})
        return _parse_candles(payload), url
    except Exception as e:
        return [], f"{url} | ERROR: {e}"


def _fetch_historical_candles(ticker: str) -> tuple[list[dict], str]:
    if not ticker:
        return [], ""
    url = f"{KALSHI_BASE}/historical/markets/{ticker}/candlesticks"
    try:
        payload = safe_get(url, params={"period_interval": 1440})
        return _parse_candles(payload), url
    except Exception as e:
        return [], f"{url} | ERROR: {e}"


def fetch_kalshi_daily_candles(markets_df: pd.DataFrame) -> None:
    print_stage("Fetching Kalshi daily candles")

    out_cols = ["market_id", "series_ticker", "Date", "kalshi_prob"]

    if markets_df.empty:
        save_csv(pd.DataFrame(columns=out_cols), "data/processed/kalshi_prices_daily.csv")
        print("No Kalshi markets available.")
        return

    if "is_relevant" in markets_df.columns:
        markets_df = markets_df[markets_df["is_relevant"] == True].copy()

    rows = []
    attempted = 0
    successful = 0
    debug_logs = 0

    for _, row in markets_df.iterrows():
        ticker = row.get("market_id")
        series_ticker = row.get("series_ticker")

        if not ticker:
            continue

        attempted += 1

        candles, live_debug = _fetch_live_candles(series_ticker, ticker)
        source = "live"

        if not candles:
            candles, hist_debug = _fetch_historical_candles(ticker)
            source = "historical"

            if debug_logs < 3:
                print(f"[KALSHI DEBUG] live attempt: {live_debug}")
                print(f"[KALSHI DEBUG] historical attempt: {hist_debug}")
                debug_logs += 1

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
        elif debug_logs < 5:
            print(f"[KALSHI DEBUG] No candles parsed for ticker={ticker} via {source}")
            debug_logs += 1

    df = pd.DataFrame(rows, columns=out_cols)
    if not df.empty:
        df = df.sort_values(["market_id", "Date"]).drop_duplicates(["market_id", "Date"])

    save_csv(df, "data/processed/kalshi_prices_daily.csv")

    print(f"Kalshi relevant markets attempted: {attempted}")
    print(f"Kalshi markets with successful candles: {successful}")
    print(f"Kalshi daily price rows: {len(df)}")


if __name__ == "__main__":
    markets = fetch_kalshi_markets()
    fetch_kalshi_daily_candles(markets)