# index_mapper.py

import pandas as pd

from country_indexes import COUNTRY_INDEXES
from seed_pairs import SEED_MARKETS
from config import (
    OUTPUT_INDEX_FEATURES,
    OUTPUT_INDEX_FEATURES_FULL,
    OUTPUT_INDEX_MAPPING,
    OUTPUT_SEED_INDEX_PANEL,
    OUTPUT_SEED_INDEX_PANEL_TS,
)


def load_index_features(path=OUTPUT_INDEX_FEATURES):
    df = pd.read_csv(path)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce", format="mixed")
    return df


def load_index_features_full(path=OUTPUT_INDEX_FEATURES_FULL):
    df = pd.read_csv(path)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce", format="mixed")
    return df


def build_country_index_df():
    return pd.DataFrame(COUNTRY_INDEXES)


def map_seed_to_countries(seed_label, theme_bucket):
    if seed_label == "ukraine_current" or theme_bucket == "geopolitics":
        return [
            "Germany", "France", "United Kingdom", "Italy", "Poland",
            "Russia", "Israel", "United States"
        ]

    if seed_label == "macro_spy" or theme_bucket == "macro":
        return [
            "United States", "Germany", "Japan", "India", "United Kingdom",
            "France", "Canada", "Australia", "China"
        ]

    if seed_label == "global_indexes" or theme_bucket == "global_macro":
        return [row["country"] for row in COUNTRY_INDEXES]

    return [row["country"] for row in COUNTRY_INDEXES]


def build_index_mapping():
    rows = []

    country_index_df = build_country_index_df()

    for seed in SEED_MARKETS:
        seed_label = seed["label"]
        theme_bucket = seed.get("theme_bucket")

        countries = map_seed_to_countries(seed_label, theme_bucket)
        mapped = country_index_df[country_index_df["country"].isin(countries)].copy()

        for _, r in mapped.iterrows():
            rows.append({
                "seed_label": seed_label,
                "theme_bucket": theme_bucket,
                "country": r["country"],
                "symbol": r["symbol"],
                "index_name": r["index_name"],
                "bucket": r.get("bucket"),
            })

    return pd.DataFrame(rows)


def build_seed_index_panel_latest():
    mapping_df = build_index_mapping()
    features_df = load_index_features()

    if mapping_df.empty or features_df.empty:
        return pd.DataFrame()

    keep_cols = [
        "symbol",
        "timestamp_utc",
        "close",
        "return_1d",
        "volatility_20d",
        "momentum_20d",
        "momentum_60d",
        "drawdown",
        "trend_50_200",
        "country",
        "index_name",
        "bucket",
    ]

    for col in keep_cols:
        if col not in features_df.columns:
            features_df[col] = None

    latest = (
        features_df.sort_values(["symbol", "timestamp_utc"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    panel = mapping_df.merge(
        latest[keep_cols],
        on=["symbol", "country", "index_name", "bucket"],
        how="left"
    )

    return panel.sort_values(["seed_label", "country", "index_name"]).reset_index(drop=True)


def build_seed_index_panel_timeseries():
    mapping_df = build_index_mapping()
    features_full_df = load_index_features_full()

    if mapping_df.empty or features_full_df.empty:
        return pd.DataFrame()

    keep_cols = [
        "symbol",
        "timestamp_utc",
        "close",
        "return_1d",
        "log_return_1d",
        "volatility_20d",
        "momentum_20d",
        "momentum_60d",
        "drawdown",
        "trend_50_200",
        "country",
        "index_name",
        "bucket",
    ]

    for col in keep_cols:
        if col not in features_full_df.columns:
            features_full_df[col] = None

    panel = mapping_df.merge(
        features_full_df[keep_cols],
        on=["symbol", "country", "index_name", "bucket"],
        how="left"
    )

    return panel.sort_values(["seed_label", "symbol", "timestamp_utc"]).reset_index(drop=True)


if __name__ == "__main__":
    mapping_df = build_index_mapping()
    panel_latest_df = build_seed_index_panel_latest()
    panel_ts_df = build_seed_index_panel_timeseries()

    mapping_df.to_csv(OUTPUT_INDEX_MAPPING, index=False)
    panel_latest_df.to_csv(OUTPUT_SEED_INDEX_PANEL, index=False)
    panel_ts_df.to_csv(OUTPUT_SEED_INDEX_PANEL_TS, index=False)

    print(mapping_df.head(30).to_string(index=False))
    print("\n--- LATEST ---")
    print(panel_latest_df.head(30).to_string(index=False))
    print("\n--- TIMESERIES ---")
    print(panel_ts_df.head(30).to_string(index=False))

    print(f"\nSaved: {OUTPUT_INDEX_MAPPING}")
    print(f"Saved: {OUTPUT_SEED_INDEX_PANEL}")
    print(f"Saved: {OUTPUT_SEED_INDEX_PANEL_TS}")