import pandas as pd

from config import POLY_GAMMA_BASE, POLY_CLOB_BASE, MAX_POLYMARKETS, START_DATE, END_DATE
from utils import save_csv, safe_get, to_unix_ts, safe_json_loads, safe_float, print_stage


def _extract_market_list(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["markets", "data", "results"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def fetch_polymarket_markets(limit: int = MAX_POLYMARKETS) -> pd.DataFrame:
    print_stage("Fetching Polymarket market metadata")
    url = f"{POLY_GAMMA_BASE}/markets"

    try:
        payload = safe_get(url, params={"limit": limit})
    except Exception as e:
        print(f"Polymarket metadata fetch failed: {e}")
        df = pd.DataFrame()
        save_csv(df, "data/processed/polymarket_markets.csv")
        return df

    markets = _extract_market_list(payload)
    rows = []

    for m in markets:
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
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    save_csv(df, "data/processed/polymarket_markets.csv")
    print_stage("Polymarket market metadata complete", len(df))
    return df


def _extract_token_ids(raw) -> list[str]:
    raw = safe_json_loads(raw)

    if raw is None:
        return []

    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]

    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            inner = cleaned.strip("[]").replace('"', "").replace("'", "")
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            return parts
        return [cleaned]

    return []


def _extract_history_rows(payload) -> list[dict]:
    if isinstance(payload, dict):
        history = payload.get("history", [])
        if isinstance(history, list):
            return history
    return []


def _parse_history_row(h: dict) -> tuple[pd.Timestamp | None, float | None]:
    ts = h.get("t")
    price = h.get("p")

    if ts is None:
        ts = h.get("timestamp")
    if price is None:
        price = h.get("price")

    if ts is None or price is None:
        return None, None

    try:
        unit = "ms" if len(str(int(ts))) >= 13 else "s"
        dt = pd.to_datetime(ts, unit=unit).floor("D")
    except Exception:
        return None, None

    price_val = safe_float(price)
    if price_val is None:
        return None, None

    if price_val > 1:
        price_val = price_val / 100.0

    return dt, price_val


def fetch_polymarket_daily_prices(markets_df: pd.DataFrame) -> None:
    print_stage("Fetching Polymarket price history")

    if markets_df.empty:
        save_csv(pd.DataFrame(), "data/processed/polymarket_prices_daily.csv")
        print("Warning: no Polymarket markets available for daily history.")
        return

    start_ts = to_unix_ts(START_DATE)
    end_ts = to_unix_ts(END_DATE)

    rows = []
    attempted_markets = 0
    successful_markets = 0

    for _, row in markets_df.iterrows():
        market_id = row.get("market_id")
        token_ids = _extract_token_ids(row.get("clob_token_ids"))

        if not market_id or not token_ids:
            continue

        attempted_markets += 1
        market_had_rows = False

        for token_id in token_ids[:2]:
            try:
                payload = safe_get(
                    f"{POLY_CLOB_BASE}/prices-history",
                    params={
                        "market": token_id,
                        "startTs": start_ts,
                        "endTs": end_ts,
                        "interval": "1d",
                    },
                )

                history_rows = _extract_history_rows(payload)
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
                    market_had_rows = True

            except Exception as e:
                print(f"Polymarket history failed for market_id={market_id}, token_id={token_id}: {e}")

        if market_had_rows:
            successful_markets += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["market_id", "Date"]).drop_duplicates(["market_id", "Date"])
    save_csv(df, "data/processed/polymarket_prices_daily.csv")

    print_stage("Polymarket price history complete", len(df))
    print(f"Polymarket attempted markets: {attempted_markets}")
    print(f"Polymarket successful markets: {successful_markets}")