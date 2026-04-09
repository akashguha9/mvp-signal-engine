# seed_tuning_report.py

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

    df["is_void"] = df["reason"].astype(str).str.startswith("void_")
    directional = df[df["signal"].isin([1, -1])].copy()

    print("=== BY SEED_LABEL ===")
    if "seed_label" not in df.columns:
        print("seed_label not found.")
        return

    by_seed = (
        df.groupby("seed_label", dropna=False)
        .agg(
            rows=("signal", "count"),
            buy_signals=("signal", lambda s: int((s == 1).sum())),
            sell_signals=("signal", lambda s: int((s == -1).sum())),
            neutral_signals=("signal", lambda s: int((s == 0).sum())),
            void_rows=("is_void", "sum"),
            acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in df.columns else ("signal", "count"),
            acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in df.columns else ("signal", "count"),
        )
        .sort_values(["rows", "acc_1d"], ascending=[False, False])
    )
    print(by_seed.to_string())

    print("\n=== DIRECTIONAL ONLY BY SEED_LABEL ===")
    if directional.empty:
        print("No directional rows.")
    else:
        by_seed_dir = (
            directional.groupby("seed_label", dropna=False)
            .agg(
                rows=("signal", "count"),
                long_rate=("signal", lambda s: float((s == 1).mean())),
                short_rate=("signal", lambda s: float((s == -1).mean())),
                acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in directional.columns else ("signal", "count"),
                acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in directional.columns else ("signal", "count"),
            )
            .sort_values(["rows", "acc_1d"], ascending=[False, False])
        )
        print(by_seed_dir.to_string())

    print("\n=== REASON x SEED_LABEL ===")
    by_reason_seed = (
        df.groupby(["seed_label", "reason"], dropna=False)
        .agg(
            rows=("signal", "count"),
            signal_mean=("signal", "mean"),
            acc_1d=("signal_success_1d", safe_mean) if "signal_success_1d" in df.columns else ("signal", "count"),
            acc_5d=("signal_success_5d", safe_mean) if "signal_success_5d" in df.columns else ("signal", "count"),
        )
        .sort_values("rows", ascending=False)
    )
    print(by_reason_seed.head(50).to_string())


if __name__ == "__main__":
    main()