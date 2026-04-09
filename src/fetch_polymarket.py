import pandas as pd

from config import POLY_GAMMA_BASE, MAX_POLYMARKETS
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
    "gta", "album", "rihanna", "playboi carti", "harvey weinstein",
    "movie", "oscar", "grammy", "celebrity", "dating", "weather",
    "nba", "nfl", "mlb", "nhl", "soccer", "football", "tennis", "golf",
    "team", "match", "game", "player", "super bowl"
}


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


if __name__ == "__main__":
    fetch_polymarket_markets()