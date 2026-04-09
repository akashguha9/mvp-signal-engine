import re
from typing import Iterable

import pandas as pd

from config import MIN_PAIR_SCORE, MAX_MATCHED_PAIRS
from utils import save_csv, print_stage


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "by", "at",
    "will", "be", "is", "are", "was", "were", "with", "from", "this", "that",
    "market", "markets", "price", "probability", "chance", "event", "events",
    "yes", "no", "than", "over", "under", "into", "after", "before", "during",
    "next", "today", "tomorrow"
}

HIGH_SIGNAL_TERMS = {
    "inflation", "fed", "fomc", "cpi", "jobs", "payrolls", "recession", "gdp",
    "election", "president", "senate", "house", "war", "ukraine", "russia",
    "china", "tariff", "sanction", "sanctions", "oil", "treasury", "rate", "rates",
    "default", "debt", "ceasefire", "iran", "israel", "bitcoin", "btc", "ethereum", "eth"
}


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    norm = _normalize_text(text)
    tokens = set(norm.split())
    tokens = {t for t in tokens if t and t not in STOPWORDS and len(t) > 2}
    return tokens


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a = set(a)
    b = set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _keyword_overlap_score(a_tokens: set[str], b_tokens: set[str]) -> float:
    a_sig = a_tokens & HIGH_SIGNAL_TERMS
    b_sig = b_tokens & HIGH_SIGNAL_TERMS
    if not a_sig or not b_sig:
        return 0.0
    inter = len(a_sig & b_sig)
    denom = max(len(a_sig | b_sig), 1)
    return inter / denom


def _date_proximity_score(poly_end, kal_close) -> float:
    if pd.isna(poly_end) or pd.isna(kal_close):
        return 0.0
    days = abs((poly_end - kal_close).days)
    if days <= 3:
        return 1.0
    if days <= 7:
        return 0.8
    if days <= 14:
        return 0.5
    if days <= 30:
        return 0.2
    return 0.0


def _pair_score(poly_question: str, kalshi_title: str, poly_end, kal_close) -> float:
    a = _tokenize(poly_question)
    b = _tokenize(kalshi_title)

    if not a or not b:
        return 0.0

    j_score = _jaccard(a, b)
    k_score = _keyword_overlap_score(a, b)
    d_score = _date_proximity_score(poly_end, kal_close)

    return 0.55 * j_score + 0.30 * k_score + 0.15 * d_score


def _safe_read_csv(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if columns:
            for c in columns:
                if c not in df.columns:
                    df[c] = None
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Missing or empty file: {path}")
        return pd.DataFrame(columns=columns or [])


def build_prediction_market_join() -> pd.DataFrame:
    print_stage("Building prediction market join")

    poly_m = _safe_read_csv(
        "data/processed/polymarket_markets.csv",
        ["market_id", "question", "category", "end_date", "is_relevant"]
    )
    poly_p = _safe_read_csv(
        "data/processed/polymarket_prices_daily.csv",
        ["market_id", "Date", "polymarket_prob"]
    )
    kalshi_m = _safe_read_csv(
        "data/processed/kalshi_markets.csv",
        ["market_id", "title", "event_ticker", "series_ticker", "close_time", "is_relevant"]
    )
    kalshi_p = _safe_read_csv(
        "data/processed/kalshi_prices_daily.csv",
        ["market_id", "Date", "kalshi_prob"]
    )

    if poly_m.empty or poly_p.empty or kalshi_m.empty or kalshi_p.empty:
        print("Warning: one or more required files are empty.")
        empty = pd.DataFrame(columns=[
            "Date", "poly_market_id", "kalshi_market_id",
            "polymarket_prob", "kalshi_prob", "belief_diff", "belief_mean",
            "poly_question", "kalshi_title", "pair_score"
        ])
        save_csv(empty, "data/processed/prediction_markets_daily_joined.csv")
        return empty

    if "is_relevant" in poly_m.columns:
        poly_m = poly_m[poly_m["is_relevant"].fillna(False) == True].copy()
    if "is_relevant" in kalshi_m.columns:
        kalshi_m = kalshi_m[kalshi_m["is_relevant"].fillna(False) == True].copy()

    poly_p["Date"] = pd.to_datetime(poly_p["Date"], errors="coerce").dt.floor("D")
    kalshi_p["Date"] = pd.to_datetime(kalshi_p["Date"], errors="coerce").dt.floor("D")
    poly_m["end_date"] = pd.to_datetime(poly_m["end_date"], errors="coerce").dt.floor("D")
    kalshi_m["close_time"] = pd.to_datetime(kalshi_m["close_time"], errors="coerce").dt.floor("D")

    poly = poly_p.merge(
        poly_m[["market_id", "question", "category", "end_date"]],
        on="market_id",
        how="inner",
    )

    kalshi = kalshi_p.merge(
        kalshi_m[["market_id", "title", "event_ticker", "series_ticker", "close_time"]],
        on="market_id",
        how="inner",
    )

    poly_meta = poly[["market_id", "question", "category", "end_date"]].drop_duplicates()
    kal_meta = kalshi[["market_id", "title", "event_ticker", "series_ticker", "close_time"]].drop_duplicates()

    print(f"Relevant Polymarket markets with prices: {len(poly_meta)}")
    print(f"Relevant Kalshi markets with prices: {len(kal_meta)}")

    candidate_pairs = []

    for _, p in poly_meta.iterrows():
        p_question = str(p.get("question") or "")
        p_end = p.get("end_date")

        if not p_question.strip():
            continue

        for _, k in kal_meta.iterrows():
            k_title = str(k.get("title") or "")
            k_close = k.get("close_time")

            if not k_title.strip():
                continue

            score = _pair_score(p_question, k_title, p_end, k_close)
            if score >= MIN_PAIR_SCORE:
                candidate_pairs.append({
                    "poly_market_id": p["market_id"],
                    "kalshi_market_id": k["market_id"],
                    "poly_question": p_question,
                    "kalshi_title": k_title,
                    "pair_score": round(score, 4),
                })

    pairs = pd.DataFrame(candidate_pairs)

    if pairs.empty:
        print("Warning: no matched Polymarket/Kalshi pairs survived scoring.")
        empty = pd.DataFrame(columns=[
            "Date", "poly_market_id", "kalshi_market_id",
            "polymarket_prob", "kalshi_prob", "belief_diff", "belief_mean",
            "poly_question", "kalshi_title", "pair_score"
        ])
        save_csv(empty, "data/processed/prediction_markets_daily_joined.csv")
        return empty

    pairs = (
        pairs.sort_values("pair_score", ascending=False)
        .drop_duplicates(subset=["poly_market_id", "kalshi_market_id"])
        .head(MAX_MATCHED_PAIRS)
    )

    print(f"Matched market pairs kept: {len(pairs)}")

    joined = pairs.merge(
        poly[["market_id", "Date", "polymarket_prob"]].rename(columns={"market_id": "poly_market_id"}),
        on="poly_market_id",
        how="left"
    ).merge(
        kalshi[["market_id", "Date", "kalshi_prob"]].rename(columns={"market_id": "kalshi_market_id"}),
        on=["kalshi_market_id", "Date"],
        how="inner"
    )

    if joined.empty:
        print("Warning: matched pairs found, but no overlapping dates.")
        save_csv(joined, "data/processed/prediction_markets_daily_joined.csv")
        return joined

    joined["belief_diff"] = joined["polymarket_prob"] - joined["kalshi_prob"]
    joined["belief_mean"] = (joined["polymarket_prob"] + joined["kalshi_prob"]) / 2.0

    joined = joined.sort_values(["Date", "pair_score"], ascending=[True, False])
    save_csv(joined, "data/processed/prediction_markets_daily_joined.csv")
    print(f"Prediction market joined rows: {len(joined)}")
    return joined


def build_final_dataset() -> None:
    print_stage("Building final dataset")

    joined = build_prediction_market_join()
    prices = _safe_read_csv(
        "data/processed/market_prices.csv",
        [
            "Date", "symbol", "price",
            "return_1d", "return_3d", "return_5d",
            "volatility_20d", "momentum_20d", "momentum_60d",
            "ma_50", "ma_200", "trend_50_200", "drawdown"
        ]
    )

    if prices.empty:
        raise RuntimeError("market_prices.csv is missing or empty.")

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.floor("D")
    prices = prices.dropna(subset=["Date"])

    if joined.empty:
        out = prices.copy()
        out["poly_market_id"] = None
        out["kalshi_market_id"] = None
        out["polymarket_prob"] = None
        out["kalshi_prob"] = None
        out["belief_diff"] = None
        out["belief_mean"] = None
        out["poly_question"] = None
        out["kalshi_title"] = None
        out["pair_score"] = None
        out["event_type"] = None
        save_csv(out, "data/processed/final_dataset.csv")
        print("Final dataset written as prices-only panel because joined belief data is empty.")
        return

    joined["Date"] = pd.to_datetime(joined["Date"], errors="coerce").dt.floor("D")
    joined["event_type"] = "macro_geopolitical"

    final_df = joined.merge(prices, on="Date", how="left")
    final_df = final_df.sort_values(["Date", "pair_score", "symbol"], ascending=[True, False, True])

    save_csv(final_df, "data/processed/final_dataset.csv")
    print(f"Final dataset rows: {len(final_df)}")


if __name__ == "__main__":
    build_final_dataset()