# build_dataset.py

import os
import glob
import pandas as pd

from config import (
    OUTPUT_EVENT_MATCHES,
    OUTPUT_EVENT_PANEL,
    OUTPUT_MODEL_DATASET,
    OUTPUT_NEWS_PANEL,
    OUTPUT_SEED_SUMMARY,
    OUTPUT_YAHOO_ENRICH,
    OUTPUT_INDEX_FEATURES,
    OUTPUT_INDEX_CORRELATIONS,
    OUTPUT_INDEX_LEADLAG,
    OUTPUT_INDEX_MAPPING,
    OUTPUT_SEED_INDEX_PANEL,
)
from seed_pairs import SEED_MARKETS


HISTORICAL_ROOT = os.path.join("historical", "news", "gdelt")


def safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def parse_mixed_datetime_col(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(
            df[col].astype(str).str.strip().replace({"nan": None, "NaT": None}),
            utc=True,
            errors="coerce",
            format="mixed",
        )
    return df


def load_historical_gdelt():
    frames = []

    for seed in SEED_MARKETS:
        seed_dir = os.path.join(HISTORICAL_ROOT, seed["label"])
        if not os.path.exists(seed_dir):
            continue

        files = sorted(glob.glob(os.path.join(seed_dir, "*.csv")))
        for f in files:
            df = safe_read_csv(f)
            if df.empty:
                continue
            df["seed_label"] = seed["label"]
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = parse_mixed_datetime_col(out, "timestamp_utc")
    return out


def build_seed_counts(seed_summary):
    if seed_summary.empty:
        return pd.DataFrame(columns=["seed_label"])

    grouped = (
        seed_summary.groupby(["seed_label", "source_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    return grouped


def build_event_match_features(event_matches):
    if event_matches.empty:
        return pd.DataFrame(columns=["seed_label"])

    df = event_matches.copy()
    df["match_score"] = pd.to_numeric(df.get("match_score"), errors="coerce")
    df["lead_lag_minutes"] = pd.to_numeric(df.get("lead_lag_minutes"), errors="coerce")

    agg = (
        df.groupby("seed_label")
        .agg(
            matched_events=("match_score", "count"),
            mean_match_score=("match_score", "mean"),
            median_match_score=("match_score", "median"),
            mean_lead_lag_minutes=("lead_lag_minutes", "mean"),
            median_lead_lag_minutes=("lead_lag_minutes", "median"),
        )
        .reset_index()
    )
    return agg


def build_historical_features(historical_gdelt):
    if historical_gdelt.empty:
        return pd.DataFrame(columns=["seed_label"])

    df = historical_gdelt.copy()

    if "timestamp_utc" not in df.columns:
        return pd.DataFrame(columns=["seed_label"])

    df = df[df["timestamp_utc"].notna()].copy()
    df["year_month"] = df["timestamp_utc"].dt.strftime("%Y-%m")

    if "headline" not in df.columns:
        df["headline"] = None
    if "source_provider" not in df.columns:
        df["source_provider"] = None

    agg = (
        df.groupby("seed_label")
        .agg(
            historical_news_rows=("headline", "count"),
            historical_unique_months=("year_month", "nunique"),
            historical_unique_sources=("source_provider", "nunique"),
        )
        .reset_index()
    )
    return agg


def build_yahoo_features(yahoo_enrich):
    if yahoo_enrich.empty:
        return pd.DataFrame(columns=["seed_label"])

    keep_cols = [
        "seed_label",
        "symbol",
        "shortName",
        "longName",
        "regularMarketPrice",
        "regularMarketChangePercent",
        "marketCap",
        "data_source",
    ]

    for col in keep_cols:
        if col not in yahoo_enrich.columns:
            yahoo_enrich[col] = None

    return yahoo_enrich[keep_cols].copy()


def build_index_seed_features(seed_index_panel):
    if seed_index_panel.empty:
        return pd.DataFrame(columns=["seed_label"])

    df = seed_index_panel.copy()

    numeric_cols = [
        "close",
        "return_1d",
        "volatility_20d",
        "momentum_20d",
        "momentum_60d",
        "drawdown",
        "trend_50_200",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "country" not in df.columns:
        df["country"] = None
    if "symbol" not in df.columns:
        df["symbol"] = None

    agg = (
        df.groupby("seed_label")
        .agg(
            mapped_indexes=("symbol", "count"),
            mapped_countries=("country", "nunique"),
            avg_index_close=("close", "mean"),
            avg_index_return_1d=("return_1d", "mean"),
            avg_index_volatility_20d=("volatility_20d", "mean"),
            avg_index_momentum_20d=("momentum_20d", "mean"),
            avg_index_momentum_60d=("momentum_60d", "mean"),
            avg_index_drawdown=("drawdown", "mean"),
            avg_index_trend_50_200=("trend_50_200", "mean"),
        )
        .reset_index()
    )
    return agg


def build_global_index_system_features(index_features, index_corr, index_leadlag):
    row = {}

    if not index_features.empty:
        row["global_index_rows"] = len(index_features)
        row["global_index_symbols"] = (
            index_features["symbol"].nunique() if "symbol" in index_features.columns else 0
        )
        row["global_index_countries"] = (
            index_features["country"].nunique() if "country" in index_features.columns else 0
        )
    else:
        row["global_index_rows"] = 0
        row["global_index_symbols"] = 0
        row["global_index_countries"] = 0

    if not index_corr.empty and "correlation" in index_corr.columns:
        corr_series = pd.to_numeric(index_corr["correlation"], errors="coerce").abs()
        row["mean_abs_index_corr"] = corr_series.mean()
        row["max_abs_index_corr"] = corr_series.max()
    else:
        row["mean_abs_index_corr"] = None
        row["max_abs_index_corr"] = None

    if not index_leadlag.empty and "best_corr" in index_leadlag.columns:
        lag_series = pd.to_numeric(index_leadlag["best_corr"], errors="coerce").abs()
        row["mean_abs_best_lag_corr"] = lag_series.mean()
        row["max_abs_best_lag_corr"] = lag_series.max()
    else:
        row["mean_abs_best_lag_corr"] = None
        row["max_abs_best_lag_corr"] = None

    return pd.DataFrame([row])


def build_dataset():
    seed_summary = safe_read_csv(OUTPUT_SEED_SUMMARY)
    event_panel = safe_read_csv(OUTPUT_EVENT_PANEL)
    news_panel = safe_read_csv(OUTPUT_NEWS_PANEL)
    event_matches = safe_read_csv(OUTPUT_EVENT_MATCHES)
    yahoo_enrich = safe_read_csv(OUTPUT_YAHOO_ENRICH)
    historical_gdelt = load_historical_gdelt()

    index_features = safe_read_csv(OUTPUT_INDEX_FEATURES)
    index_corr = safe_read_csv(OUTPUT_INDEX_CORRELATIONS)
    index_leadlag = safe_read_csv(OUTPUT_INDEX_LEADLAG)
    index_mapping = safe_read_csv(OUTPUT_INDEX_MAPPING)
    seed_index_panel = safe_read_csv(OUTPUT_SEED_INDEX_PANEL)

    seed_counts = build_seed_counts(seed_summary)
    event_features = build_event_match_features(event_matches)
    historical_features = build_historical_features(historical_gdelt)
    yahoo_features = build_yahoo_features(yahoo_enrich)
    seed_index_features = build_index_seed_features(seed_index_panel)
    global_index_features = build_global_index_system_features(
        index_features, index_corr, index_leadlag
    )

    dataset = pd.DataFrame({
        "seed_label": [seed["label"] for seed in SEED_MARKETS],
        "theme_bucket": [seed.get("theme_bucket") for seed in SEED_MARKETS],
        "asset_class": [seed.get("asset_class") for seed in SEED_MARKETS],
        "historical_start": [seed.get("historical_start") for seed in SEED_MARKETS],
    })

    for frame in [
        seed_counts,
        event_features,
        historical_features,
        yahoo_features,
        seed_index_features,
    ]:
        if not frame.empty:
            dataset = dataset.merge(frame, on="seed_label", how="left")

    dataset["current_event_panel_rows"] = len(event_panel) if not event_panel.empty else 0
    dataset["current_news_panel_rows"] = len(news_panel) if not news_panel.empty else 0
    dataset["current_index_mapping_rows"] = len(index_mapping) if not index_mapping.empty else 0
    dataset["current_seed_index_panel_rows"] = len(seed_index_panel) if not seed_index_panel.empty else 0

    if not global_index_features.empty:
        for col in global_index_features.columns:
            dataset[col] = global_index_features.iloc[0][col]

    # optional nice column ordering
    preferred_order = [
        "seed_label",
        "theme_bucket",
        "asset_class",
        "historical_start",
        "market_news",
        "polymarket_market",
        "kalshi_market",
        "matched_events",
        "mean_match_score",
        "median_match_score",
        "mean_lead_lag_minutes",
        "median_lead_lag_minutes",
        "historical_news_rows",
        "historical_unique_months",
        "historical_unique_sources",
        "symbol",
        "shortName",
        "longName",
        "regularMarketPrice",
        "regularMarketChangePercent",
        "marketCap",
        "data_source",
        "mapped_indexes",
        "mapped_countries",
        "avg_index_close",
        "avg_index_return_1d",
        "avg_index_volatility_20d",
        "avg_index_momentum_20d",
        "avg_index_momentum_60d",
        "avg_index_drawdown",
        "avg_index_trend_50_200",
        "current_event_panel_rows",
        "current_news_panel_rows",
        "current_index_mapping_rows",
        "current_seed_index_panel_rows",
        "global_index_rows",
        "global_index_symbols",
        "global_index_countries",
        "mean_abs_index_corr",
        "max_abs_index_corr",
        "mean_abs_best_lag_corr",
        "max_abs_best_lag_corr",
    ]

    ordered_cols = [c for c in preferred_order if c in dataset.columns]
    remaining_cols = [c for c in dataset.columns if c not in ordered_cols]
    dataset = dataset[ordered_cols + remaining_cols]

    return dataset


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv(OUTPUT_MODEL_DATASET, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {OUTPUT_MODEL_DATASET}")