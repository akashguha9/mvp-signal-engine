# rank_signal_engine.py

from __future__ import annotations

import numpy as np
import pandas as pd


INPUT_FILE = "output_timeseries_dataset_dedup.csv"
OUTPUT_FILE = "output_rank_signals.csv"


FEATURES = [
    "t0_momentum_20d",
    "t0_momentum_60d",
    "t0_trend_50_200",
    "t0_volatility_20d",
    "t0_drawdown",
]


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Missing file: {path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Empty file: {path}")
        return pd.DataFrame()

    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce", format="mixed")

    numeric_cols = FEATURES + [
        "match_score",
        "lead_lag_minutes",
        "future_return_1d",
        "future_return_3d",
        "future_return_5d",
        "future_return_20d",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 5-day oriented score:
    # - flatter trend is better
    # - less negative 20d/60d momentum is better
    # - moderate volatility is better
    # - moderate drawdown is better
    #
    # Designed after dedup to avoid clustered-event illusions.

    out["score_trend_flat"] = -out["t0_trend_50_200"].abs()
    out["score_mom20_less_negative"] = out["t0_momentum_20d"]
    out["score_mom60_less_negative"] = out["t0_momentum_60d"]
    out["score_vol_mid"] = -(out["t0_volatility_20d"] - 0.014).abs()
    out["score_dd_mid"] = -(out["t0_drawdown"] + 0.085).abs()

    out["rank_score_raw"] = (
        0.30 * out["score_trend_flat"].fillna(0)
        + 0.25 * out["score_mom20_less_negative"].fillna(0)
        + 0.20 * out["score_mom60_less_negative"].fillna(0)
        + 0.15 * out["score_vol_mid"].fillna(0)
        + 0.10 * out["score_dd_mid"].fillna(0)
    )

    return out


def rank_within_day(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "event_time" not in out.columns:
        out["rank_pct"] = np.nan
        return out

    out["event_day"] = out["event_time"].dt.floor("D")
    out["rank_pct"] = (
        out.groupby("event_day")["rank_score_raw"]
        .rank(method="average", pct=True)
    )
    return out


def assign_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Top 5% only
    long_mask = out["rank_pct"] >= 0.95

    out["signal"] = 0
    out["reason"] = "rank_neutral"

    out.loc[long_mask, "signal"] = 1
    out.loc[long_mask, "reason"] = "rank_top_5pct_long"

    return out


def evaluate_signal(row: pd.Series, horizon: int = 5):
    col = f"future_return_{horizon}d"
    signal = row.get("signal", 0)

    if col not in row.index:
        return np.nan

    val = row.get(col)
    if pd.isna(val):
        return np.nan

    if signal == 1:
        return int(val > 0)
    if signal == -1:
        return int(val < 0)

    return np.nan


def summarize_directional(df: pd.DataFrame) -> None:
    directional = df[df["signal"] != 0].copy()

    print(f"\ndirectional rows: {len(directional)}")
    if directional.empty:
        return

    for h in [1, 3, 5]:
        col = f"signal_success_{h}d"
        if col in directional.columns:
            print(f"{col}: {directional[col].mean()}")

    # daily bucket view for the traded tail
    if "event_day" in directional.columns:
        day_summary = (
            directional.groupby("event_day")
            .agg(
                rows=("signal", "count"),
                acc_5d=("signal_success_5d", "mean"),
            )
            .reset_index()
        )

        print("\nDaily traded-tail summary:")
        print(day_summary.head(20).to_string(index=False))


def main():
    print("=== RANK SIGNAL ENGINE (5D) ===")
    print(f"input_file: {INPUT_FILE}")

    df = load_data(INPUT_FILE)
    if df.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False)
        print("No data.")
        return

    df = add_scores(df)
    df = rank_within_day(df)
    df = assign_signals(df)

    for h in [1, 3, 5]:
        col = f"future_return_{h}d"
        if col in df.columns:
            df[f"signal_success_{h}d"] = df.apply(lambda r: evaluate_signal(r, horizon=h), axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"rows: {len(df)}")
    print("\nSignal counts:")
    print(df["signal"].value_counts(dropna=False).to_string())

    print("\nReason counts:")
    print(df["reason"].value_counts(dropna=False).to_string())

    summarize_directional(df)

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()