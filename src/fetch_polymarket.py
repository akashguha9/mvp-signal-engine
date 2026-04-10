import pandas as pd

from config import POLY_GAMMA_BASE, POLY_CLOB_BASE, MAX_POLYMARKETS
from utils import save_csv, safe_get, safe_json_loads, safe_float, print_stage


ALLOW_TERMS = {
    "inflation", "fed", "fomc", "cpi", "ppi", "jobs", "payrolls", "unemployment",
    "recession", "gdp", "yield", "treasury", "rates", "rate cut", "rate hike",
    "election", "president", "senate", "house", "white house",
    "war", "ukraine", "russia", "china", "taiwan", "iran", "israel", "nato",
    "tariff", "sanction", "sanctions", "ceasefire", "oil", "crude", "opec",
    "bitcoin", "btc", "ethereum", "eth", "sec", "etf"
}

BLOCK_TERMS = {
    "gta", "album", "rihanna", "playboi", "carti", "harvey weinstein",
    "movie", "oscar", "grammy", "celebrity", "dating", "weather",
    "nba", "nfl", "mlb", "nhl", "soccer", "football", "tennis", "golf",
    "team", "match", "game", "player", "super bowl"
}

# SAFE CAP FOR NOW
MAX_RELEVANT_MARKETS_TO_FETCH = 15


def _extract_market_list(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["markets", "data", "results"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _is_relevant_market(question: str, description: str, category: str) -> bool:
    text = " ".join([_norm(question), _norm(description), _norm(category)])
    if any(term in text for term in BLOCK_TERMS):
        return False
    return any(term in text for term in ALLOW_TERMS)


def _extract_token_ids(raw) -> list[str]:
    raw = safe_json_loads(raw)

    if raw is None:
        return []

    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]

    if isinstance(raw, str):
        cleaned = raw.strip()

        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned.replace("'", '"')
            parsed = safe_json_loads(cleaned)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]

            inner = cleaned.strip("[]").replace('"', "").replace("'", "")
            return [p.strip() for p in inner.split(",") if p.strip()]

        return [cleaned]

    return []


def _extract_history_rows(payload) -> list[dict]:
    if isinstance(payload, dict):
        for key in ["history", "data", "pricesHistory", "prices_history"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    if isinstance(payload, list):
        return payload
    return []


def _parse_history_row(h: dict) -> tuple[pd.Timestamp | None, float | None]:
    ts = h.get("t")
    price = h.get("p")

    if ts is None:
        ts = h.get("timestamp")
    if price is None:
        price = h.get("price")
    if price is None:
        price = h.get("y")

    if ts is None or price is None:
        return None, None

    try:
        unit = "ms" if len(str(int(ts))) >= 13 else "s"
        dt = pd.to_datetime(ts, unit=unit).floor("D")
    except Exception:
        return None, None

    prob = safe_float(price)
    if prob is None:
        return None, None

    if prob > 1:
        prob = prob / 100.0

    return dt, prob


def fetch_polymarket_markets(limit: int = MAX_POLYMARKETS) -> pd.DataFrame:
    print_stage("Fetching Polymarket market metadata")

    url = f"{POLY_GAMMA_BASE}/markets"
    try:
        payload = safe_get(url, params={"limit": limit})
    except Exception as e:
        print(f"Polymarket metadata fetch failed: {e}")
        df = pd.DataFrame(columns=[
            "market_id", "question", "description", "category", "active", "closed",
            "end_date", "slug", "outcomes", "condition_id", "clob_token_ids", "is_relevant"
        ])
        save_csv(df, "data/processed/polymarket_markets.csv")
        return df

    markets = _extract_market_list(payload)
    rows = []

    for m in markets:
        question = m.get("question")
        description = m.get("description")
        category = m.get("category")

        rows.append({
            "market_id": m.get("id"),
            "question": question,
            "description": description,
            "category": category,
            "active": m.get("active"),
            "closed": m.get("closed"),
            "end_date": m.get("endDate"),
            "slug": m.get("slug"),
            "outcomes": m.get("outcomes"),
            "condition_id": m.get("conditionId"),
            "clob_token_ids": m.get("clobTokenIds"),
            "is_relevant": _is_relevant_market(question, description, category),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    save_csv(df, "data/processed/polymarket_markets.csv")

    total = len(df)
    relevant = int(df["is_relevant"].fillna(False).sum()) if not df.empty else 0
    print(f"Polymarket total markets fetched: {total}")
    print(f"Polymarket relevant markets kept: {relevant}")

    if relevant > 0:
        print("\nTop relevant Polymarket markets preview:")
        print(df[df["is_relevant"] == True][["market_id", "question", "category"]].head(20).to_string())

    return df


def fetch_polymarket_daily_prices(markets_df: pd.DataFrame) -> None:
    print_stage("Fetching Polymarket daily prices")

    out_cols = ["market_id", "poly_token_id", "Date", "polymarket_prob"]

    if markets_df.empty:
        save_csv(pd.DataFrame(columns=out_cols), "data/processed/polymarket_prices_daily.csv")
        print("No Polymarket metadata available.")
        return

    if "is_relevant" in markets_df.columns:
        markets_df = markets_df[markets_df["is_relevant"] == True].copy()

    markets_df = markets_df.head(MAX_RELEVANT_MARKETS_TO_FETCH).copy()

    rows = []
    attempted = 0
    success = 0

    for _, row in markets_df.iterrows():
        market_id = row.get("market_id")
        token_ids = _extract_token_ids(row.get("clob_token_ids"))

        if not market_id or not token_ids:
            continue

        token_id = str(token_ids[0])
        attempted += 1

        try:
            import time
            now = int(time.time())
            start = now - 30 * 24 * 3600

            payload = safe_get(
                f"{POLY_CLOB_BASE}/prices-history",
                params={
                    "market": token_id,
                    "startTs": start,
                    "endTs": now,
                    "interval": "1d",
                },
            )

            history_rows = _extract_history_rows(payload)

            if history_rows:
                success += 1

            for h in history_rows:
                dt, prob = _parse_history_row(h)
                if dt is None or prob is None:
                    continue

                rows.append({
                    "market_id": market_id,
                    "poly_token_id": token_id,
                    "Date": dt,
                    "polymarket_prob": prob,
                })

            print(f"Fetched market_id={market_id} | token_id={token_id[:18]}... | history_rows={len(history_rows)}")

        except Exception as e:
            print(f"[POLY DEBUG] market_id={market_id} failed: {e}")

    df = pd.DataFrame(rows, columns=out_cols)

    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.floor("D")
        df = df.dropna(subset=["Date", "polymarket_prob"])
        df = (
            df.sort_values(["market_id", "Date", "poly_token_id"])
              .groupby(["market_id", "poly_token_id", "Date"], as_index=False)["polymarket_prob"]
              .last()
        )

    save_csv(df, "data/processed/polymarket_prices_daily.csv")

    print(f"Polymarket relevant markets attempted: {attempted}")
    print(f"Polymarket markets with successful history: {success}")
    print(f"Polymarket daily price rows: {len(df)}")


if __name__ == "__main__":
    markets = fetch_polymarket_markets()
    fetch_polymarket_daily_prices(markets)