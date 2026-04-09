# index_mapper.py

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    OUTPUT_SEED_INDEX_PANEL,
    OUTPUT_SEED_INDEX_PANEL_TS,
)


def load_seed_index_panel(path: str = OUTPUT_SEED_INDEX_PANEL) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )

    numeric_cols = [
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def ensure_normalized_trend(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp_utc").copy()

    # if trend_50_200 is missing or obviously broken, rebuild from ma_50 / ma_200
    if "ma_50" in g.columns and "ma_200" in g.columns:
        need_rebuild = (
            "trend_50_200" not in g.columns
            or g["trend_50_200"].dropna().empty
            or g["trend_50_200"].dropna().abs().median() > 1
        )

        if need_rebuild:
            g["trend_50_200"] = (g["ma_50"] / g["ma_200"]) - 1

    return g


def build_seed_index_panel_ts() -> pd.DataFrame:
    df = load_seed_index_panel()

    if df.empty:
        return pd.DataFrame()

    required = ["seed_label", "symbol", "timestamp_utc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in OUTPUT_SEED_INDEX_PANEL: {missing}")

    df = df[df["timestamp_utc"].notna()].copy()

    parts = []
    for (_, _), g in df.groupby(["seed_label", "symbol"], dropna=False, sort=False):
        g2 = ensure_normalized_trend(g)
        parts.append(g2)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if not out.empty:
        out = out.sort_values(["seed_label", "symbol", "timestamp_utc"]).reset_index(drop=True)

    return out


if __name__ == "__main__":
    out = build_seed_index_panel_ts()
    out.to_csv(OUTPUT_SEED_INDEX_PANEL_TS, index=False)

    print(f"rows: {len(out)}")
    print(out.columns.tolist())
    print(out.head(20).to_string(index=False))

    if "trend_50_200" in out.columns:
        print("\ntrend_50_200 summary:")
        print(out["trend_50_200"].describe().to_string())

    print(f"\nSaved: {OUTPUT_SEED_INDEX_PANEL_TS}")