import pandas as pd

from utils import save_csv


KEYWORDS = [
    "inflation", "fed", "rates", "cpi", "jobs", "recession",
    "election", "war", "ukraine", "russia", "china", "oil",
    "gdp", "tariff", "sanction", "default", "treasury"
]


def _text_match(a: str, b: str) -> bool:
    a = (a or "").lower()
    b = (b or "").lower()
    if not a or not b:
        return False
    return any(k in a and k in b for k in KEYWORDS)


def build_prediction_market_join() -> pd.DataFrame:
    poly_m = pd.read_csv("data/processed/polymarket_markets.csv")
    poly_p = pd.read_csv("data/processed/polymarket_prices_daily.csv")
    kalshi_m = pd.read_csv("data/processed/kalshi_markets.csv")
    kalshi_p = pd.read_csv("data/processed/kalshi_prices_daily.csv")

    poly_p["Date"] = pd.to_datetime(poly_p["Date"]).dt.floor("D")
    kalshi_p["Date"] = pd.to_datetime(kalshi_p["Date"]).dt.floor("D")

    poly = poly_p.merge(
        poly_m[["market_id", "question", "category", "end_date"]],
        on="market_id",
        how="left",
    )

    kalshi = kalshi_p.merge(
        kalshi_m[["market_id", "title", "event_ticker", "series_ticker", "close_time"]],
        on="market_id",
        how="left",
    )

    candidate_pairs = []
    poly_meta = poly[["market_id", "question", "category"]].drop_duplicates()
    kal_meta = kalshi[["market_id", "title", "event_ticker", "series_ticker"]].drop_duplicates()

    for _, p in poly_meta.iterrows():
        for _, k in kal_meta.iterrows():
            if _text_match(str(p.get("question")), str(k.get("title"))):
                candidate_pairs.append({
                    "poly_market_id": p["market_id"],
                    "kalshi_market_id": k["market_id"],
                    "poly_question": p.get("question"),
                    "kalshi_title": k.get("title"),
                })

    pairs = pd.DataFrame(candidate_pairs)
    if pairs.empty:
        print("Warning: no Polymarket/Kalshi market pairs matched by keywords.")
        empty = pd.DataFrame(columns=[
            "Date", "poly_market_id", "kalshi_market_id", "polymarket_prob", "kalshi_prob",
            "belief_diff", "belief_mean", "poly_question", "kalshi_title"
        ])
        save_csv(empty, "data/processed/prediction_markets_daily_joined.csv")
        return empty

    joined = pairs.merge(
        poly[["market_id", "Date", "polymarket_prob"]].rename(columns={"market_id": "poly_market_id"}),
        on="poly_market_id",
        how="left"
    ).merge(
        kalshi[["market_id", "Date", "kalshi_prob"]].rename(columns={"market_id": "kalshi_market_id"}),
        on=["kalshi_market_id", "Date"],
        how="inner"
    )

    joined["belief_diff"] = joined["polymarket_prob"] - joined["kalshi_prob"]
    joined["belief_mean"] = (joined["polymarket_prob"] + joined["kalshi_prob"]) / 2.0

    save_csv(joined, "data/processed/prediction_markets_daily_joined.csv")
    return joined


def build_final_dataset() -> None:
    joined = build_prediction_market_join()
    prices = pd.read_csv("data/processed/market_prices.csv")
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.floor("D")

    if joined.empty:
        out = prices.copy()
        out["poly_market_id"] = None
        out["kalshi_market_id"] = None
        out["polymarket_prob"] = None
        out["kalshi_prob"] = None
        out["belief_diff"] = None
        out["belief_mean"] = None
        out["event_type"] = None
        save_csv(out, "data/processed/final_dataset.csv")
        return

    joined["event_type"] = "macro_geopolitical"

    # Cross-join joined belief data with each asset day on the same date
    final_df = joined.merge(prices, on="Date", how="left")

    save_csv(final_df, "data/processed/final_dataset.csv")


if __name__ == "__main__":
    build_final_dataset()