# polymarket.py

import pandas as pd

from config import POLY_GAMMA_BASE
from helpers import safe_get_json


def fetch_polymarket_markets(limit=1000, closed=True):
    params = {
        "limit": limit,
        "closed": "true" if closed else "false"
    }

    data = safe_get_json(f"{POLY_GAMMA_BASE}/markets", params=params)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    rows = []
    for m in data:
        rows.append({
            "market_id": m.get("id"),
            "slug": m.get("slug"),
            "question": m.get("question") or m.get("title"),
            "description": m.get("description"),
            "start_date": m.get("startDate") or m.get("start_date"),
            "end_date": m.get("endDate") or m.get("end_date"),
            "created_at": m.get("createdAt") or m.get("created_at"),
            "updated_at": m.get("updatedAt") or m.get("updated_at"),
            "active": m.get("active"),
            "closed": m.get("closed"),
            "liquidity": m.get("liquidity"),
            "volume": m.get("volume"),
            "resolved_outcome": m.get("outcome") or m.get("result"),
            "clob_token_ids": m.get("clobTokenIds"),
        })

    return pd.DataFrame(rows)


def load_polymarket_universe(limit=1000):
    open_df = fetch_polymarket_markets(limit=limit, closed=False)
    closed_df = fetch_polymarket_markets(limit=limit, closed=True)

    frames = []
    if not open_df.empty:
        frames.append(open_df)
    if not closed_df.empty:
        frames.append(closed_df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["market_id"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = load_polymarket_universe(limit=200)
    print(df.head(20).to_string(index=False))
    print(f"\nRows: {len(df)}")