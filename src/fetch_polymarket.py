import pandas as pd

from config import POLY_GAMMA_BASE, POLY_CLOB_BASE, MAX_POLYMARKETS, START_DATE, END_DATE
from utils import save_csv, safe_get, to_unix_ts, safe_json_loads, safe_float, print_stage


RELEVANT_TERMS = {
    "inflation", "fed", "fomc", "cpi", "jobs", "payrolls", "recession", "gdp",
    "election", "president", "senate", "house", "war", "ukraine", "russia",
    "china", "tariff", "sanction", "sanctions", "oil", "treasury", "rate", "rates",
    "default", "debt", "ceasefire", "iran", "israel", "bitcoin", "btc", "ethereum", "eth"
}

# Polymarket rejects very long windows, so fetch in chunks.
CHUNK_DAYS = 60
CHUNK_SECONDS = CHUNK_DAYS * 24 * 3600


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
    return any(term in text for term in RELEVANT_TERMS)


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
        for key in ["history", "data", "pricesHistory", "prices_history"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
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


def _fetch_history_chunk(token_id: str, chunk_start: int, chunk_end: int):
    return safe_get(
        f"{POLY_CLOB_BASE}/prices-history",
        params={
            "market": token_id,
            "startTs": chunk_start,
            "endTs": chunk_end,
            "interval": "1d",
        },
    )


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

    start_ts = to_unix_ts(START_DATE)
    end_ts = to_unix_ts(END_DATE)

    rows = []
    attempted_markets = 0
    token_markets = 0
    successful_markets = 0
    successful_chunks = 0
    debug_failures = 0

    for _, row in markets_df.iterrows():
        market_id = row.get("market_id")
        token_ids = _extract_token_ids(row.get("clob_token_ids"))

        if not market_id:
            continue

        attempted_markets += 1

        if not token_ids:
            continue

        token_markets += 1
        market_had_rows = False

        # Usually first 1-2 token IDs are enough for testing/building
        for token_id in token_ids[:2]:
            current_start = start_ts

            while current_start < end_ts:
                current_end = min(current_start + CHUNK_SECONDS, end_ts)

                try:
                    payload = _fetch_history_chunk(token_id, current_start, current_end)
                    history_rows = _extract_history_rows(payload)

                    if history_rows:
                        successful_chunks += 1

                    if not history_rows and debug_failures < 3:
                        print(
                            f"[POLY DEBUG] Empty history | market_id={market_id} | token_id={token_id} | "
                            f"chunk_start={current_start} | chunk_end={current_end}"
                        )
                        if isinstance(payload, dict):
                            print(f"[POLY DEBUG] Payload keys: {list(payload.keys())}")
                        else:
                            print(f"[POLY DEBUG] Payload type: {type(payload)}")
                        debug_failures += 1

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
                    msg = str(e)
                    if debug_failures < 5:
                        print(
                            f"[POLY DEBUG] Chunk fetch failed | market_id={market_id} | token_id={token_id} | "
                            f"chunk_start={current_start} | chunk_end={current_end} | error={msg}"
                        )
                        debug_failures += 1

                    # If a chunk is still too large for some edge case, halve it once.
                    if "interval is too long" in msg.lower():
                        smaller_end = min(current_start + (CHUNK_SECONDS // 2), end_ts)
                        if smaller_end > current_start:
                            try:
                                payload = _fetch_history_chunk(token_id, current_start, smaller_end)
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
                            except Exception as inner_e:
                                if debug_failures < 6:
                                    print(
                                        f"[POLY DEBUG] Retry failed | market_id={market_id} | token_id={token_id} | "
                                        f"error={inner_e}"
                                    )
                                    debug_failures += 1

                current_start = current_end

        if market_had_rows:
            successful_markets += 1

    df = pd.DataFrame(rows, columns=out_cols)

    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.floor("D")
        df = df.dropna(subset=["Date", "polymarket_prob"])
        df = (
            df.sort_values(["market_id", "Date", "poly_token_id"])
              .drop_duplicates(["market_id", "Date"], keep="first")
        )

    save_csv(df, "data/processed/polymarket_prices_daily.csv")

    print(f"Polymarket relevant markets attempted: {attempted_markets}")
    print(f"Polymarket markets with token IDs: {token_markets}")
    print(f"Polymarket markets with successful history: {successful_markets}")
    print(f"Polymarket successful chunks: {successful_chunks}")
    print(f"Polymarket daily price rows: {len(df)}")


if __name__ == "__main__":
    markets = fetch_polymarket_markets()
    fetch_polymarket_daily_prices(markets)