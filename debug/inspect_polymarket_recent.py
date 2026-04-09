# inspect_polymarket_recent.py

import pandas as pd
from polymarket import fetch_polymarket_markets

START_DATE = "2025-01-01"
END_DATE = pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def prepare(df, label):
    if df.empty:
        return df

    out = df.copy()
    out["end_date_dt"] = pd.to_datetime(out["end_date"], utc=True, errors="coerce")

    start_ts = pd.to_datetime(START_DATE, utc=True)
    end_ts = pd.to_datetime(END_DATE, utc=True)

    out = out[out["end_date_dt"].notna()].copy()
    out = out[(out["end_date_dt"] >= start_ts) & (out["end_date_dt"] <= end_ts)].copy()
    out["source_bucket"] = label

    keep_cols = [
        "source_bucket",
        "market_id",
        "question",
        "end_date",
        "slug",
        "volume",
        "resolved_outcome",
    ]

    for col in keep_cols:
        if col not in out.columns:
            out[col] = None

    return out[keep_cols].sort_values("end_date", ascending=False).reset_index(drop=True)


closed_df = fetch_polymarket_markets(limit=1000, closed=True)
open_df = fetch_polymarket_markets(limit=1000, closed=False)

closed_recent = prepare(closed_df, "closed")
open_recent = prepare(open_df, "open")

print("\nCLOSED RECENT MARKETS")
print("=" * 80)
if closed_recent.empty:
    print("No closed recent markets found.")
else:
    print(closed_recent.head(100).to_string(index=False))

print("\nOPEN RECENT MARKETS")
print("=" * 80)
if open_recent.empty:
    print("No open recent markets found.")
else:
    print(open_recent.head(100).to_string(index=False))