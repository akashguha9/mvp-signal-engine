# debug_polymarket_history.py

import pandas as pd
from polymarket import fetch_polymarket_markets, fetch_polymarket_price_history

START_DATE = "2021-01-01"
END_DATE = "2021-12-31"

KEYWORDS = ["inflation", "coinbase", "bnb", "ethereum", "trump", "airbnb"]


def contains_any_keyword(text, keywords):
    if not text:
        return False
    t = text.lower()
    return any(k.lower() in t for k in keywords)


markets = fetch_polymarket_markets(limit=300, closed=True)

markets = markets[
    markets["question"].fillna("").apply(lambda x: contains_any_keyword(x, KEYWORDS))
].copy()

print("Matched Polymarket markets:")
if markets.empty:
    print("No matched markets found.")
    raise SystemExit

print(markets[["market_id", "question", "end_date", "clob_token_ids"]].head(10).to_string(index=False))

first_market_id = markets.iloc[0]["market_id"]
first_question = markets.iloc[0]["question"]
first_clob_token_ids = markets.iloc[0]["clob_token_ids"]

print("\nTesting price history for:")
print(first_market_id, "-", first_question)
print("clob_token_ids =", first_clob_token_ids)

hist = fetch_polymarket_price_history(
    market_id=first_market_id,
    start_date=START_DATE,
    end_date=END_DATE,
    clob_token_ids=first_clob_token_ids,
    debug=True
)

print(f"\nHistory rows: {len(hist)}")

if hist.empty:
    print("No history returned.")
else:
    print(hist.head(20).to_string(index=False))
    print("\nMin timestamp:", hist["timestamp_utc"].min())
    print("Max timestamp:", hist["timestamp_utc"].max())