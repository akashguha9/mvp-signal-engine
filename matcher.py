# matcher.py

import pandas as pd

from config import (
    DEFAULT_SIGNAL_START_DATE,
    FINNHUB_API_KEY,
    OUTPUT_EVENT_PANEL,
    OUTPUT_NEWS_PANEL,
    OUTPUT_PRICE_PANEL,
    OUTPUT_SEED_SUMMARY,
)
from polymarket import load_polymarket_universe
from kalshi import fetch_kalshi_historical_markets, fetch_kalshi_candles
from market_news import fetch_market_news
from seed_pairs import SEED_MARKETS


SIGNAL_START_DATE = DEFAULT_SIGNAL_START_DATE
SIGNAL_END_DATE = pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def contains_any_keyword(text, keywords):
    if not text:
        return False
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def filter_polymarket_by_keywords(poly_df, keywords, start_date, end_date):
    if poly_df.empty:
        return poly_df.copy()

    df = poly_df.copy()
    keyword_mask = df["question"].fillna("").apply(lambda x: contains_any_keyword(x, keywords))

    # use end_date for recency filter, but later use created_at/start_date for event timing
    df["end_date_dt"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)

    date_mask = (
        df["end_date_dt"].notna()
        & (df["end_date_dt"] >= start_ts)
        & (df["end_date_dt"] <= end_ts)
    )

    return df[keyword_mask & date_mask].copy().reset_index(drop=True)


def fetch_seeded_kalshi_markets(kalshi_tickers):
    if not kalshi_tickers:
        return pd.DataFrame(columns=[
            "ticker", "title", "subtitle", "status",
            "result", "close_time", "expiration_time"
        ])

    df, _ = fetch_kalshi_historical_markets(
        limit=100,
        cursor=None,
        exclude_sports=False
    )

    if df.empty:
        return df

    return df[df["ticker"].isin(kalshi_tickers)].copy().reset_index(drop=True)


def build_seed_summary(seed, poly_df, kalshi_df, news_df):
    rows = []

    for _, p in poly_df.iterrows():
        rows.append({
            "seed_label": seed["label"],
            "source_type": "polymarket_market",
            "source_id": p.get("market_id"),
            "title": p.get("question"),
            "timestamp_utc": p.get("created_at") or p.get("start_date") or p.get("end_date"),
            "extra": p.get("slug")
        })

    for _, k in kalshi_df.iterrows():
        rows.append({
            "seed_label": seed["label"],
            "source_type": "kalshi_market",
            "source_id": k.get("ticker"),
            "title": k.get("title"),
            "timestamp_utc": k.get("close_time"),
            "extra": k.get("result")
        })

    for _, n in news_df.iterrows():
        rows.append({
            "seed_label": seed["label"],
            "source_type": "market_news",
            "source_id": n.get("symbol_or_query"),
            "title": n.get("headline"),
            "timestamp_utc": n.get("timestamp_utc"),
            "extra": n.get("source_provider")
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce", format="mixed")
    out = out[out["timestamp_utc"].notna()].copy()
    return out.sort_values(["timestamp_utc", "source_type"]).reset_index(drop=True)


def build_kalshi_price_panel(seed, kalshi_df):
    panels = []

    for _, k in kalshi_df.iterrows():
        hist = fetch_kalshi_candles(
            ticker=k["ticker"],
            start_date=SIGNAL_START_DATE,
            end_date=SIGNAL_END_DATE
        ).copy()

        if not hist.empty:
            hist["seed_label"] = seed["label"]
            hist["platform"] = "kalshi"
            hist["market_ref"] = k["ticker"]
            hist["market_title"] = k["title"]
            panels.append(hist)

    if not panels:
        return pd.DataFrame(columns=[
            "timestamp_utc", "price_yes", "seed_label",
            "platform", "market_ref", "market_title"
        ])

    out = pd.concat(panels, ignore_index=True)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce", format="mixed")
    out = out[out["timestamp_utc"].notna()].copy()
    return out.sort_values(["timestamp_utc", "platform"]).reset_index(drop=True)


def build_polymarket_metadata_panel(seed, poly_df):
    if poly_df.empty:
        return pd.DataFrame(columns=[
            "timestamp_utc", "platform", "price_yes", "seed_label",
            "market_ref", "market_title", "headline", "source_provider"
        ])

    out = poly_df.copy()
    out["timestamp_utc"] = pd.to_datetime(
        out["created_at"].fillna(out["start_date"]).fillna(out["end_date"]),
        utc=True,
        errors="coerce",
        format="mixed"
    )
    out = out[out["timestamp_utc"].notna()].copy()

    out["platform"] = "polymarket_market"
    out["price_yes"] = None
    out["seed_label"] = seed["label"]
    out["market_ref"] = out["market_id"]
    out["market_title"] = out["question"]
    out["headline"] = out["question"]
    out["source_provider"] = "polymarket"

    return out[[
        "timestamp_utc", "platform", "price_yes", "seed_label",
        "market_ref", "market_title", "headline", "source_provider"
    ]].sort_values("timestamp_utc").reset_index(drop=True)


def build_news_panel(seed, news_df):
    if news_df.empty:
        return pd.DataFrame(columns=[
            "timestamp_utc", "platform", "price_yes", "seed_label",
            "market_ref", "market_title", "headline", "source_provider"
        ])

    out = news_df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce", format="mixed")
    out = out[out["timestamp_utc"].notna()].copy()

    out["platform"] = "market_news"
    out["price_yes"] = None
    out["seed_label"] = seed["label"]
    out["market_ref"] = out["symbol_or_query"]
    out["market_title"] = out["headline"]

    return out[[
        "timestamp_utc", "platform", "price_yes", "seed_label",
        "market_ref", "market_title", "headline", "source_provider"
    ]].sort_values("timestamp_utc").reset_index(drop=True)


def build_full_event_panel(polymarket_meta_df, kalshi_price_df, news_panel_df):
    if not kalshi_price_df.empty:
        kalshi_price_df = kalshi_price_df.copy()
        kalshi_price_df["headline"] = None
        kalshi_price_df["source_provider"] = "kalshi"
        kalshi_price_df = kalshi_price_df[[
            "timestamp_utc", "platform", "price_yes", "seed_label",
            "market_ref", "market_title", "headline", "source_provider"
        ]]

    frames = []
    if not polymarket_meta_df.empty:
        frames.append(polymarket_meta_df)
    if not kalshi_price_df.empty:
        frames.append(kalshi_price_df)
    if not news_panel_df.empty:
        frames.append(news_panel_df)

    if not frames:
        return pd.DataFrame(columns=[
            "timestamp_utc", "platform", "price_yes", "seed_label",
            "market_ref", "market_title", "headline", "source_provider"
        ])

    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp_utc"] = pd.to_datetime(panel["timestamp_utc"], utc=True, errors="coerce", format="mixed")
    panel = panel[panel["timestamp_utc"].notna()].copy()
    return panel.sort_values(["timestamp_utc", "platform"]).reset_index(drop=True)


def get_api_key_for_seed(seed):
    if seed["asset_class"] == "stock_etf":
        return FINNHUB_API_KEY
    return ""


if __name__ == "__main__":
    print("Fetching Polymarket markets...")
    poly_all = load_polymarket_universe(limit=1000)
    print(f"Polymarket combined rows fetched: {len(poly_all)}")

    all_seed_summaries = []
    all_kalshi_price_panels = []
    all_news_panels = []
    all_event_panels = []

    for seed in SEED_MARKETS:
        print("\n" + "=" * 70)
        print(f"SEED: {seed['label']}")
        print("=" * 70)

        poly_df = filter_polymarket_by_keywords(
            poly_all,
            seed["polymarket_keywords"],
            SIGNAL_START_DATE,
            SIGNAL_END_DATE
        )
        print(f"Polymarket metadata matches for seed '{seed['label']}': {len(poly_df)}")

        kalshi_df = fetch_seeded_kalshi_markets(seed["kalshi_tickers"])
        print(f"Kalshi seeded matches for '{seed['label']}': {len(kalshi_df)}")

        try:
            news_df = fetch_market_news(
                asset_class=seed["asset_class"],
                api_key=get_api_key_for_seed(seed),
                symbol_or_query=seed["news_symbol_or_query"],
                start_date=SIGNAL_START_DATE,
                end_date=SIGNAL_END_DATE,
                extra_params=seed.get("news_extra_params", {}),
            )
            print(f"News rows for '{seed['label']}': {len(news_df)}")
        except Exception as e:
            print(f"News fetch failed for '{seed['label']}': {e}")
            news_df = pd.DataFrame(columns=[
                "asset_class", "source_provider", "symbol_or_query",
                "headline", "snippet", "article_url", "timestamp_utc",
                "sentiment", "raw"
            ])

        summary_df = build_seed_summary(seed, poly_df, kalshi_df, news_df)
        kalshi_price_df = build_kalshi_price_panel(seed, kalshi_df)
        polymarket_meta_df = build_polymarket_metadata_panel(seed, poly_df)
        news_panel_df = build_news_panel(seed, news_df)
        event_panel_df = build_full_event_panel(polymarket_meta_df, kalshi_price_df, news_panel_df)

        all_seed_summaries.append(summary_df)
        all_kalshi_price_panels.append(kalshi_price_df)
        all_news_panels.append(news_panel_df)
        all_event_panels.append(event_panel_df)

    final_summary = pd.concat(all_seed_summaries, ignore_index=True) if all_seed_summaries else pd.DataFrame()
    final_prices = pd.concat(all_kalshi_price_panels, ignore_index=True) if all_kalshi_price_panels else pd.DataFrame()
    final_news = pd.concat(all_news_panels, ignore_index=True) if all_news_panels else pd.DataFrame()
    final_panel = pd.concat(all_event_panels, ignore_index=True) if all_event_panels else pd.DataFrame()

    final_summary.to_csv(OUTPUT_SEED_SUMMARY, index=False)
    final_prices.to_csv(OUTPUT_PRICE_PANEL, index=False)
    final_news.to_csv(OUTPUT_NEWS_PANEL, index=False)
    final_panel.to_csv(OUTPUT_EVENT_PANEL, index=False)

    print("\nSaved files:")
    print(f"- {OUTPUT_SEED_SUMMARY}")
    print(f"- {OUTPUT_PRICE_PANEL}")
    print(f"- {OUTPUT_NEWS_PANEL}")
    print(f"- {OUTPUT_EVENT_PANEL}")