# debug_seed_match.py

import pandas as pd
from polymarket import fetch_polymarket_markets
from seed_pairs import SEED_MARKETS

SIGNAL_START_DATE = "2025-01-01"
SIGNAL_END_DATE = pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def contains_any_keyword(text, keywords):
    if not text:
        return False
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def filter_polymarket_by_keywords(poly_df, keywords, start_date, end_date):
    if poly_df.empty:
        return poly_df.copy()

    df = poly_df.copy()

    keyword_mask = df["question"].fillna("").apply(
        lambda x: contains_any_keyword(x, keywords)
    )

    df["end_date_dt"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)

    date_mask = (
        df["end_date_dt"].notna()
        & (df["end_date_dt"] >= start_ts)
        & (df["end_date_dt"] <= end_ts)
    )

    out = df[keyword_mask & date_mask].copy().reset_index(drop=True)
    return out


poly_open = fetch_polymarket_markets(limit=1000, closed=False)
poly_closed = fetch_polymarket_markets(limit=1000, closed=True)

poly_all = pd.concat([poly_open, poly_closed], ignore_index=True).drop_duplicates(subset=["market_id"])

print(f"Open rows: {len(poly_open)}")
print(f"Closed rows: {len(poly_closed)}")
print(f"Combined unique rows: {len(poly_all)}")

seed = SEED_MARKETS[0]
print("\nSeed:")
print(seed)

matched = filter_polymarket_by_keywords(
    poly_all,
    seed["polymarket_keywords"],
    SIGNAL_START_DATE,
    SIGNAL_END_DATE
)

print(f"\nMatched rows: {len(matched)}")

if matched.empty:
    print("\nNo current Polymarket matches for this seed.")
else:
    print("\nMatched current Polymarket rows:")
    print(
        matched[["market_id", "question", "end_date", "slug", "volume"]]
        .sort_values("end_date", ascending=False)
        .head(50)
        .to_string(index=False)
    )