import pandas as pd

from config import POLY_GAMMA_BASE, POLY_CLOB_BASE, MAX_POLYMARKETS, START_DATE, END_DATE
from utils import save_csv, safe_get, to_unix_ts


def fetch_polymarket_markets(limit: int = MAX_POLYMARKETS) -> pd.DataFrame:
    print("Fetching Polymarket market metadata...")
    url = f"{POLY_GAMMA_BASE}/markets"
    data = safe_get(url, params={"limit": limit})

    rows = []
    for m in data if isinstance(data, list) else []:
        rows.append({
            "market_id": m.get("id"),
            "question": m.get("question"),
            "description": m.get("description"),
            "category": m.get("category"),
            "active": m.get("active"),
            "closed": m.get("closed"),
            "end_date": m.get("endDate"),
            "slug": m.get("slug"),
            "outcomes": m.get("outcomes"),
            "condition_id": m.get("conditionId"),
            "clob_token_ids": m.get("clobTokenIds"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        save_csv(df, "data/processed/polymarket_markets.csv")
    else:
        print("Warning: Polymarket market metadata empty.")
    return df


def _extract_first_token_id(raw) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, list) and raw:
        return str(raw[0])
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned.strip("[]").replace('"', "").replace("'", "")
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            return parts[0] if parts else None
        return cleaned
    return None


def fetch_polymarket_daily_prices(markets_df: pd.DataFrame) -> None:
    print("Fetching Polymarket price history...")
    start_ts = to_unix_ts(START_DATE)
    end_ts = to_unix_ts(END_DATE)

    rows = []
    for _, row in markets_df.iterrows():
        token_id = _extract_first_token_id(row.get("clob_token_ids"))
        market_id = row.get("market_id")

        if not token_id:
            continue

        try:
            url = f"{POLY_CLOB_BASE}/prices-history"
            payload = safe_get(url, params={
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "interval": "1d",
            })

            history = payload.get("history", [])
            for h in history:
                ts = h.get("t")
                price = h.get("p")
                if ts is None or price is None:
                    continue
                rows.append({
                    "market_id": market_id,
                    "poly_token_id": token_id,
                    "Date": pd.to_datetime(ts, unit="s").floor("D"),
                    "polymarket_prob": float(price),
                })
        except Exception as e:
            print(f"Polymarket history failed for market_id={market_id}, token_id={token_id}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["market_id", "Date"]).drop_duplicates(["market_id", "Date"])
        save_csv(df, "data/processed/polymarket_prices_daily.csv")
    else:
        print("Warning: no Polymarket daily prices fetched.")


if __name__ == "__main__":
    mdf = fetch_polymarket_markets()
    fetch_polymarket_daily_prices(mdf)