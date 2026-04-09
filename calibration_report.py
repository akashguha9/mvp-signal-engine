# calibration_report.py

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
        print("No signal rows found.")
        return

    df["is_void"] = df["reason"].astype(str).str.startswith("void_")

    print("=== GLOBAL SUMMARY ===")
    print(f"rows: {len(df)}")
    print(f"void rows: {int(df['is_void'].sum())}")
    print(f"non-void rows: {int((~df['is_void']).sum())}")
    print("\nSignal counts:")
    print(df["signal"].value_counts(dropna=False).to_string())

    print("\nOverall accuracy:")
    for col in ["signal_success_1d", "signal_success_5d", "signal_success_20d"]:
        if col in df.columns:
            print(f"{col}: {safe_mean(df[col])}")

    print("\n=== VOID VS NON-VOID ===")
    parts = []
    for label, sub in [("VOID", df[df["is_void"]]), ("NON_VOID", df[~df["is_void"]])]:
        row = {
            "segment": label,
            "rows": len(sub),
            "buy_signals": int((sub["signal"] == 1).sum()),
            "sell_signals": int((sub["signal"] == -1).sum()),
            "neutral_signals": int((sub["signal"] == 0).sum()),
            "acc_1d": safe_mean(sub["signal_success_1d"]) if "signal_success_1d" in sub.columns else np.nan,
            "acc_5d": safe_mean(sub["signal_success_5d"]) if "signal_success_5d" in sub.columns else np.nan,
            "acc_20d": safe_mean(sub["signal_success_20d"]) if "signal_success_20d" in sub.columns else np.nan,
        }
        parts.append(row)

    print(pd.DataFrame(parts).to_string(index=False))

    print("\n=== BY REASON ===")
    by_reason = (
        df.groupby("reason", dropna=False)
        .agg(
            rows=("signal", "count"),
            signal_mean=("signal", "mean"),
            acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in df.columns else ("signal", "count"),
            acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in df.columns else ("signal", "count"),
        )
        .sort_values("rows", ascending=False)
    )
    print(by_reason.head(30).to_string())

    if "seed_label" in df.columns:
        print("\n=== BY SEED_LABEL ===")
        by_seed = (
            df.groupby("seed_label", dropna=False)
            .agg(
                rows=("signal", "count"),
                buy_signals=("signal", lambda s: int((s == 1).sum())),
                sell_signals=("signal", lambda s: int((s == -1).sum())),
                neutral_signals=("signal", lambda s: int((s == 0).sum())),
                acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in df.columns else ("signal", "count"),
                acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in df.columns else ("signal", "count"),
            )
            .sort_values("rows", ascending=False)
        )
        print(by_seed.to_string())

    print("\n=== DIRECTIONAL ACCURACY ===")
    directional = df[df["signal"].isin([1, -1])].copy()
    if directional.empty:
        print("No directional rows.")
    else:
        by_dir = (
            directional.groupby("signal")
            .agg(
                rows=("signal", "count"),
                acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in directional.columns else ("signal", "count"),
                acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in directional.columns else ("signal", "count"),
            )
            .sort_index()
        )
        print(by_dir.to_string())

    print("\n=== USABLE SAMPLE COUNTS ===")
    usable = {}
    for col in ["signal_success_1d", "signal_success_5d", "signal_success_20d"]:
        if col in df.columns:
            usable[col] = int(df[col].notna().sum())
    print(pd.Series(usable).to_string())


if __name__ == "__main__":
    main()