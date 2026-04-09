# oos_validation.py

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
        print("No rows found.")
        return

    # pick best available time column
    time_col = None
    for c in ["event_time", "news_time", "polymarket_time", "t0_timestamp"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        print("No usable time column found.")
        return

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce", format="mixed")
    df = df[df[time_col].notna()].copy().sort_values(time_col).reset_index(drop=True)

    if df.empty:
        print("No rows with valid timestamps.")
        return

    df["is_void"] = df["reason"].astype(str).str.startswith("void_")

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print("=== OUT-OF-SAMPLE VALIDATION ===")
    print(f"time column: {time_col}")
    print(f"train rows: {len(train)}")
    print(f"test rows:  {len(test)}")

    if not train.empty:
        print(f"train range: {train[time_col].min()} -> {train[time_col].max()}")
    if not test.empty:
        print(f"test range:  {test[time_col].min()} -> {test[time_col].max()}")

    def summarize(name: str, x: pd.DataFrame):
        print(f"\n=== {name} ===")
        print(f"rows: {len(x)}")
        print(f"void rows: {int(x['is_void'].sum())}")
        print("signal counts:")
        print(x["signal"].value_counts(dropna=False).to_string())

        for col in ["signal_success_1d", "signal_success_5d", "signal_success_20d"]:
            if col in x.columns:
                print(f"{col}: {safe_mean(x[col])}")

        directional = x[x["signal"].isin([1, -1])].copy()
        print(f"directional rows: {len(directional)}")
        if not directional.empty:
            for col in ["signal_success_1d", "signal_success_5d", "signal_success_20d"]:
                if col in directional.columns:
                    print(f"{col} (directional only): {safe_mean(directional[col])}")

        if "seed_label" in x.columns:
            print("\nTop seeds:")
            top_seed = (
                x.groupby("seed_label")
                .agg(
                    rows=("signal", "count"),
                    signal_mean=("signal", "mean"),
                    acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in x.columns else ("signal", "count"),
                    acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in x.columns else ("signal", "count"),
                )
                .sort_values("rows", ascending=False)
                .head(15)
            )
            print(top_seed.to_string())

        print("\nTop reasons:")
        top_reason = (
            x.groupby("reason")
            .agg(
                rows=("signal", "count"),
                signal_mean=("signal", "mean"),
                acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in x.columns else ("signal", "count"),
                acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in x.columns else ("signal", "count"),
            )
            .sort_values("rows", ascending=False)
            .head(20)
        )
        print(top_reason.to_string())

    summarize("TRAIN", train)
    summarize("TEST", test)

    print("\n=== DELTA CHECK ===")
    for col in ["signal_success_1d", "signal_success_5d"]:
        if col in df.columns:
            train_acc = safe_mean(train[col])
            test_acc = safe_mean(test[col])
            delta = test_acc - train_acc if pd.notna(train_acc) and pd.notna(test_acc) else np.nan
            print(f"{col}: train={train_acc}, test={test_acc}, delta={delta}")


if __name__ == "__main__":
    main()