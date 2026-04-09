# signal_engine.py

import numpy as np
import pandas as pd

from config import OUTPUT_TIMESERIES_DATASET, OUTPUT_SIGNALS, MIN_SIGNAL_MATCH_SCORE
from data_void_engine import infer_signal_with_void_fallback


def load_timeseries_dataset(path=OUTPUT_TIMESERIES_DATASET):
    df = pd.read_csv(path)

    datetime_cols = [
        "event_time",
        "polymarket_time",
        "news_time",
        "t0_timestamp",
        "t1d_timestamp",
        "t5d_timestamp",
        "t20d_timestamp",
        "symbol_earliest_timestamp",
        "symbol_latest_timestamp",
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                utc=True,
                errors="coerce",
                format="mixed",
            )

    numeric_cols = [
        "match_score",
        "lead_lag_minutes",
        "t0_close",
        "t0_return_1d",
        "t0_volatility_20d",
        "t0_momentum_20d",
        "t0_momentum_60d",
        "t0_drawdown",
        "t0_trend_50_200",
        "future_return_1d",
        "future_return_5d",
        "future_return_20d",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    coverage_cols = [c for c in df.columns if c.startswith("has_") and c.endswith("_forward")]
    for col in coverage_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def infer_signal(row, df_history=None):
    return infer_signal_with_void_fallback(
        row=row,
        df_history=df_history,
        min_score=MIN_SIGNAL_MATCH_SCORE,
        force_direction=False,
    )


def evaluate_signal(row, horizon=5):
    signal = row.get("signal", 0)
    fwd_col = f"future_return_{horizon}d"

    if fwd_col not in row.index:
        return np.nan

    fwd = row.get(fwd_col)

    if pd.isna(fwd):
        return np.nan

    if signal == 1:
        return int(fwd > 0)
    if signal == -1:
        return int(fwd < 0)

    return np.nan


def build_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "seed_label" not in df.columns:
        return pd.DataFrame()

    agg_map = {
        "rows": ("signal", "count"),
        "buy_signals": ("signal", lambda s: int((s == 1).sum())),
        "sell_signals": ("signal", lambda s: int((s == -1).sum())),
        "neutral_signals": ("signal", lambda s: int((s == 0).sum())),
    }

    if "match_score" in df.columns:
        agg_map["mean_match_score"] = ("match_score", "mean")

    if "future_return_1d" in df.columns:
        agg_map["mean_future_return_1d"] = ("future_return_1d", "mean")
    if "future_return_5d" in df.columns:
        agg_map["mean_future_return_5d"] = ("future_return_5d", "mean")
    if "future_return_20d" in df.columns:
        agg_map["mean_future_return_20d"] = ("future_return_20d", "mean")

    if "signal_success_1d" in df.columns:
        agg_map["signal_accuracy_1d"] = ("signal_success_1d", "mean")
    if "signal_success_5d" in df.columns:
        agg_map["signal_accuracy_5d"] = ("signal_success_5d", "mean")
    if "signal_success_20d" in df.columns:
        agg_map["signal_accuracy_20d"] = ("signal_success_20d", "mean")

    if "has_1d_forward" in df.columns:
        agg_map["coverage_1d"] = ("has_1d_forward", "mean")
    if "has_5d_forward" in df.columns:
        agg_map["coverage_5d"] = ("has_5d_forward", "mean")
    if "has_20d_forward" in df.columns:
        agg_map["coverage_20d"] = ("has_20d_forward", "mean")

    if "reason" in df.columns:
        agg_map["void_rows"] = ("reason", lambda s: int(s.astype(str).str.startswith("void_").sum()))

    summary = (
        df.groupby(["seed_label"])
        .agg(**agg_map)
        .reset_index()
    )

    return summary


if __name__ == "__main__":
    df = load_timeseries_dataset()

    if df.empty:
        print("No timeseries dataset rows.")
        pd.DataFrame().to_csv(OUTPUT_SIGNALS, index=False)
    else:
        signals = []
        reasons = []

        for _, row in df.iterrows():
            signal, reason = infer_signal(row, df_history=df)
            signals.append(signal)
            reasons.append(reason)

        df["signal"] = signals
        df["reason"] = reasons

        df["signal_success_1d"] = df.apply(lambda r: evaluate_signal(r, horizon=1), axis=1)
        df["signal_success_5d"] = df.apply(lambda r: evaluate_signal(r, horizon=5), axis=1)
        df["signal_success_20d"] = df.apply(lambda r: evaluate_signal(r, horizon=20), axis=1)

        summary = build_signal_summary(df)

        df.to_csv(OUTPUT_SIGNALS, index=False)

        print("Signal summary:")
        if not summary.empty:
            print(summary.to_string(index=False))
        else:
            print("No summary available.")

        if "reason" in df.columns:
            print("\nReason counts:")
            print(df["reason"].value_counts(dropna=False).head(25).to_string())

        print(f"\nSaved: {OUTPUT_SIGNALS}")