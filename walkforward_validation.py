# walkforward_validation.py

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from config import OUTPUT_SIGNALS


def safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


def evaluate_block(df: pd.DataFrame, label: str):
    print(f"\n=== {label} ===")
    print(f"rows: {len(df)}")

    if df.empty:
        return

    print("signal counts:")
    print(df["signal"].value_counts(dropna=False).to_string())

    for col in ["signal_success_1d", "signal_success_5d", "signal_success_20d"]:
        if col in df.columns:
            print(f"{col}: {safe_mean(df[col])}")

    directional = df[df["signal"].isin([1, -1])].copy()
    print(f"directional rows: {len(directional)}")

    if not directional.empty:
        for col in ["signal_success_1d", "signal_success_5d", "signal_success_20d"]:
            if col in directional.columns:
                print(f"{col} (directional): {safe_mean(directional[col])}")

    if "reason" in df.columns:
        print("\nTop reasons:")
        top_reasons = (
            df.groupby("reason", dropna=False)
            .agg(
                rows=("signal", "count"),
                signal_mean=("signal", "mean"),
                acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in df.columns else ("signal", "count"),
                acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in df.columns else ("signal", "count"),
            )
            .sort_values("rows", ascending=False)
            .head(15)
        )
        print(top_reasons.to_string())

    if "seed_label" in df.columns:
        print("\nTop seeds:")
        top_seeds = (
            df.groupby("seed_label", dropna=False)
            .agg(
                rows=("signal", "count"),
                signal_mean=("signal", "mean"),
                acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in df.columns else ("signal", "count"),
                acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in df.columns else ("signal", "count"),
            )
            .sort_values("rows", ascending=False)
            .head(15)
        )
        print(top_seeds.to_string())


def main():
    df = pd.read_csv(OUTPUT_SIGNALS)

    if df.empty:
        print("No data.")
        return

    time_col = None
    for c in ["event_time", "news_time", "polymarket_time", "t0_timestamp"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        print("No time column found.")
        return

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce", format="mixed")
    df = df[df[time_col].notna()].sort_values(time_col).reset_index(drop=True)

    if df.empty:
        print("No rows with valid timestamps.")
        return

    print("=== WALK-FORWARD VALIDATION ===")
    print(f"time column: {time_col}")
    print(f"total rows: {len(df)}")
    print(f"range: {df[time_col].min()} -> {df[time_col].max()}")

    splits = [0.6, 0.7, 0.8, 0.9]

    for split in splits:
        split_idx = int(len(df) * split)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()

        print("\n" + "=" * 60)
        print(f"SPLIT: {int(split * 100)} / {int((1 - split) * 100)}")

        if not train.empty:
            print(f"train range: {train[time_col].min()} -> {train[time_col].max()}")
        if not test.empty:
            print(f"test range:  {test[time_col].min()} -> {test[time_col].max()}")

        evaluate_block(train, "TRAIN")
        evaluate_block(test, "TEST")

        print("\nDELTA CHECK")
        for col in ["signal_success_1d", "signal_success_5d", "signal_success_20d"]:
            if col in df.columns:
                train_acc = safe_mean(train[col])
                test_acc = safe_mean(test[col])

                if pd.notna(train_acc) and pd.notna(test_acc):
                    print(f"{col}: train={train_acc}, test={test_acc}, delta={test_acc - train_acc}")
                else:
                    print(f"{col}: train={train_acc}, test={test_acc}, delta=nan")

        train_dir = train[train["signal"].isin([1, -1])].copy()
        test_dir = test[test["signal"].isin([1, -1])].copy()

        print("\nDIRECTIONAL DELTA CHECK")
        for col in ["signal_success_1d", "signal_success_5d", "signal_success_20d"]:
            if col in df.columns:
                train_acc = safe_mean(train_dir[col]) if not train_dir.empty else np.nan
                test_acc = safe_mean(test_dir[col]) if not test_dir.empty else np.nan

                if pd.notna(train_acc) and pd.notna(test_acc):
                    print(f"{col} directional: train={train_acc}, test={test_acc}, delta={test_acc - train_acc}")
                else:
                    print(f"{col} directional: train={train_acc}, test={test_acc}, delta=nan")

        print("=" * 60)


if __name__ == "__main__":
    main()