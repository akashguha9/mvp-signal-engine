import pandas as pd

from config import KALSHI_BASE, MAX_KALSHI_MARKETS
from utils import save_csv, safe_get, print_stage


ALLOW_TERMS = {
    "inflation", "fed", "fomc", "cpi", "ppi", "jobs", "payrolls", "unemployment",
    "recession", "gdp", "yield", "treasury", "rates", "rate cut", "rate hike",
    "election", "president", "senate", "house", "white house",
    "war", "ukraine", "russia", "china", "taiwan", "iran", "israel", "nato",
    "tariff", "sanction", "sanctions", "ceasefire", "oil", "crude", "opec",
    "bitcoin", "btc", "ethereum", "eth", "sec", "etf"
}

BLOCK_TERMS = {
    "nba", "nfl", "mlb", "nhl", "soccer", "football", "tennis", "golf",
    "match", "game", "player", "championship", "album", "movie", "oscar",
    "grammy", "celebrity", "weather"
}


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _is_relevant_market(title: str, subtitle: str, event_ticker: str, series_ticker: str) -> bool:
    text = " ".join([_norm(title), _norm(subtitle), _norm(event_ticker), _norm(series_ticker)])

    if any(term in text for term in BLOCK_TERMS):
        return False

    return any(term in text for term in ALLOW_TERMS)


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

    if relevant > 0:
        print("\nTop relevant Kalshi markets preview:")
        print(df[df["is_relevant"] == True][["market_id", "title", "series_ticker", "event_ticker"]].head(20).to_string())

    return df


if __name__ == "__main__":
    fetch_kalshi_markets()