# leakage_audit.py

from __future__ import annotations

import numpy as np
import pandas as pd

from config import OUTPUT_SIGNALS


def safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


def main():
    df = pd.read_csv(OUTPUT_SIGNALS)

    if df.empty:
        print("No data.")
        return

    time_cols = ["event_time", "news_time", "polymarket_time", "t0_timestamp", "t1d_timestamp", "t5d_timestamp"]
    for c in time_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="mixed")

    print("=== LEAKAGE AUDIT ===")

    if "event_time" in df.columns and "t0_timestamp" in df.columns:
        bad_t0 = df[df["t0_timestamp"] > df["event_time"]]
        print(f"rows where t0_timestamp > event_time: {len(bad_t0)}")
        if not bad_t0.empty:
            cols = [c for c in ["seed_label", "event_time", "t0_timestamp", "reason", "signal"] if c in bad_t0.columns]
            print(bad_t0[cols].head(20).to_string(index=False))

    if "t0_timestamp" in df.columns and "t1d_timestamp" in df.columns:
        bad_t1 = df[df["t1d_timestamp"] < df["t0_timestamp"]]
        print(f"\nrows where t1d_timestamp < t0_timestamp: {len(bad_t1)}")

    if "t0_timestamp" in df.columns and "t5d_timestamp" in df.columns:
        bad_t5 = df[df["t5d_timestamp"] < df["t0_timestamp"]]
        print(f"rows where t5d_timestamp < t0_timestamp: {len(bad_t5)}")

    if "event_time" in df.columns:
        df = df.sort_values("event_time").reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        test = df.iloc[split_idx:].copy()

        print("\n=== TEST REGIME PROFILE ===")
        cols = [c for c in ["t0_trend_50_200", "t0_momentum_20d", "t0_momentum_60d", "t0_drawdown", "t0_volatility_20d"] if c in test.columns]
        if cols:
            print(test[cols].describe().to_string())

        if "signal" in test.columns:
            print("\nTest signal counts:")
            print(test["signal"].value_counts(dropna=False).to_string())

        if "reason" in test.columns:
            print("\nTop test reasons:")
            print(test["reason"].astype(str).value_counts().head(20).to_string())

    directional = df[df["signal"].isin([1, -1])].copy()
    if not directional.empty and "reason" in directional.columns:
        print("\n=== DIRECTIONAL BY REASON ===")
        out = (
            directional.groupby("reason")
            .agg(
                rows=("signal", "count"),
                acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in directional.columns else ("signal", "count"),
                acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in directional.columns else ("signal", "count"),
            )
            .sort_values("rows", ascending=False)
        )
        print(out.head(25).to_string())

    if "seed_label" in directional.columns:
        print("\n=== DIRECTIONAL BY SEED ===")
        out2 = (
            directional.groupby("seed_label")
            .agg(
                rows=("signal", "count"),
                long_rate=("signal", lambda s: float((s == 1).mean())),
                short_rate=("signal", lambda s: float((s == -1).mean())),
                acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in directional.columns else ("signal", "count"),
                acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in directional.columns else ("signal", "count"),
            )
            .sort_values("rows", ascending=False)
        )
        print(out2.to_string())


if __name__ == "__main__":
    main()