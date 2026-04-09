# rolling_validation_strict.py

from __future__ import annotations

import sys
import numpy as np
import pandas as pd


def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "output_signals.csv"

    df = pd.read_csv(input_file)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce", format="mixed")
    df = df[df["event_time"].notna()].sort_values("event_time").reset_index(drop=True)

    window_size = 300
    step = 150
    min_directional = 50

    print("=== STRICT ROLLING VALIDATION ===")
    print(f"input_file: {input_file}")

    kept = []
    dropped = []

    for start in range(0, len(df) - window_size, step):
        end = start + window_size
        test = df.iloc[start:end].copy()
        directional = test[test["signal"].isin([1, -1])].copy()
        acc = safe_mean(directional["signal_success_1d"])

        row = {
            "start": start,
            "end": end,
            "rows": len(test),
            "directional": len(directional),
            "acc_1d": acc,
        }

        if len(directional) >= min_directional:
            kept.append(row)
        else:
            dropped.append(row)

    print("\n=== KEPT WINDOWS ===")
    for r in kept:
        print(r)

    print("\n=== DROPPED WINDOWS (< min directional) ===")
    for r in dropped:
        print(r)

    if kept:
        accs = [r["acc_1d"] for r in kept if pd.notna(r["acc_1d"])]
        print("\n=== KEPT SUMMARY ===")
        print("count:", len(accs))
        print("mean:", np.nanmean(accs))
        print("std:", np.nanstd(accs))
        print("min:", np.nanmin(accs))
        print("max:", np.nanmax(accs))
    else:
        print("\nNo kept windows.")


if __name__ == "__main__":
    main()