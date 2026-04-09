# build_seed_index_panel.py

from __future__ import annotations

import pandas as pd

from config import (
    OUTPUT_INDEX_MAPPING,
    OUTPUT_INDEX_FEATURES_FULL,
    OUTPUT_SEED_INDEX_PANEL,
)


def load_mapping(path: str = OUTPUT_INDEX_MAPPING) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize columns
    if "symbol" not in df.columns:
        raise ValueError("OUTPUT_INDEX_MAPPING must contain column: symbol")
    if "seed_label" not in df.columns:
        raise ValueError("OUTPUT_INDEX_MAPPING must contain column: seed_label")

    return df


def load_features(path: str = OUTPUT_INDEX_FEATURES_FULL) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "symbol" not in df.columns:
        raise ValueError("OUTPUT_INDEX_FEATURES_FULL must contain column: symbol")
    if "timestamp_utc" not in df.columns:
        raise ValueError("OUTPUT_INDEX_FEATURES_FULL must contain column: timestamp_utc")

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )

    return df


def build_seed_index_panel() -> pd.DataFrame:
    mapping = load_mapping()
    features = load_features()

    # keep only rows with usable time
    features = features[features["timestamp_utc"].notna()].copy()

    # merge historical feature rows onto every mapped seed/symbol
    out = mapping.merge(features, on="symbol", how="inner")

    # keep useful columns first if they exist
    preferred_cols = [
        "seed_label",
        "theme_bucket",
        "country",
        "symbol",
        "index_name",
        "bucket",
        "timestamp_utc",
        "close",
        "return_1d",
        "log_return_1d",
        "volatility_20d",
        "momentum_20d",
        "momentum_60d",
        "drawdown",
        "ma_20",
        "ma_50",
        "ma_200",
        "trend_50_200",
    ]

    existing_preferred = [c for c in preferred_cols if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_preferred]
    out = out[existing_preferred + remaining]

    out = out.sort_values(["seed_label", "symbol", "timestamp_utc"]).reset_index(drop=True)

    return out


if __name__ == "__main__":
    out = build_seed_index_panel()
    out.to_csv(OUTPUT_SEED_INDEX_PANEL, index=False)

    print(f"rows: {len(out)}")
    print(out.columns.tolist())
    print(out.head(20).to_string(index=False))

    if "timestamp_utc" in out.columns:
        print("\nTime range:")
        print(out["timestamp_utc"].min(), "->", out["timestamp_utc"].max())

    if "seed_label" in out.columns:
        print("\nRows by seed:")
        print(out.groupby("seed_label").size().to_string())

    print(f"\nSaved: {OUTPUT_SEED_INDEX_PANEL}")