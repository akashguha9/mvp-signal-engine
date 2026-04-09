# signal_engine.py

from __future__ import annotations

import numpy as np
import pandas as pd

import data_void_engine
from config import OUTPUT_SIGNALS, MIN_SIGNAL_MATCH_SCORE
from data_void_engine import infer_signal_with_void_fallback


print("USING signal_engine FROM:", __file__)
print("USING data_void_engine FROM:", data_void_engine.__file__)


def load_timeseries_dataset(path="output_timeseries_dataset_dedup.csv"):
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    datetime_cols = [
        "event_time",
        "polymarket_time",
        "news_time",
        "t0_timestamp",
        "t1d_timestamp",
        "t3d_timestamp",
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
        "t0_log_return_1d",
        "t0_volatility_20d",
        "t0_momentum_20d",
        "t0_momentum_60d",
        "t0_drawdown",
        "t0_trend_50_200",
        "future_return_1d",
        "future_return_3d",
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


def detect_regime(row: pd.Series) -> str:
    trend = row.get("t0_trend_50_200")
    vol = row.get("t0_volatility_20d")
    mom20 = row.get("t0_momentum_20d")
    drawdown = row.get("t0_drawdown")

    if pd.isna(trend) or pd.isna(vol):
        return "unknown"

    if trend > 0.02 and pd.notna(mom20) and mom20 > 0:
        return "strong_uptrend"

    if trend < -0.02 and pd.notna(mom20) and mom20 < 0:
        return "downtrend"

    if vol > 0.02:
        return "high_vol"

    if pd.notna(drawdown) and drawdown < -0.10:
        return "drawdown_stress"

    return "mixed"


def base_reason(reason: str) -> str:
    if pd.isna(reason):
        return ""
    return str(reason).split("|")[0]


def infer_signal(row, df_history=None):
    """
    Recalibrated baseline:
    - use upstream strict data_void_engine
    - keep one confirmed long rule family in strong_uptrend
    - add a regime transition rule for raw neutral rows
    - everything else becomes neutral
    """

    raw_signal, raw_reason = infer_signal_with_void_fallback(
        row=row,
        df_history=df_history,
        min_score=MIN_SIGNAL_MATCH_SCORE,
        force_direction=False,
    )

    regime = detect_regime(row)
    reason_root = base_reason(raw_reason)

    # REGIME TRANSITION SIGNAL
    transition_signal = (
        row.get("t0_trend_50_200", 0) > -0.01
        and row.get("t0_trend_50_200", 0) < 0.01
        and row.get("t0_momentum_20d", 0) > -0.02
        and row.get("t0_momentum_60d", 0) > -0.06
        and row.get("t0_volatility_20d", 0) > 0.012
        and row.get("t0_drawdown", 0) > -0.12
    )

    # TIME SPACING FILTER (anti-cluster)
    recent_event = (
        hasattr(row, "event_time")
        and pd.notna(row.get("event_time"))
    )

    time_filter = True
    if recent_event:
        try:
            ts = pd.to_datetime(row["event_time"], utc=True)
            time_filter = (ts.minute % 30 == 0)
        except Exception:
            time_filter = True

    # ENTRY
    if raw_signal == 0 and transition_signal and time_filter:
        return 1, f"regime_transition|regime_{regime}"

    if raw_signal == 0:
        return 0, f"neutral_{reason_root}|regime_{regime}"

    keep_long = (
        raw_signal == 1
        and regime == "strong_uptrend"
        and reason_root == "matched_tiebreak_up_uptrend_mom_up"
    )

    if keep_long:
        return 1, f"{reason_root}|regime_{regime}"

    return 0, f"filtered_out_{reason_root}|regime_{regime}"


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
    if "t0_trend_50_200" in df.columns:
        agg_map["mean_t0_trend_50_200"] = ("t0_trend_50_200", "mean")
    if "t0_momentum_20d" in df.columns:
        agg_map["mean_t0_momentum_20d"] = ("t0_momentum_20d", "mean")
    if "t0_momentum_60d" in df.columns:
        agg_map["mean_t0_momentum_60d"] = ("t0_momentum_60d", "mean")
    if "t0_volatility_20d" in df.columns:
        agg_map["mean_t0_volatility_20d"] = ("t0_volatility_20d", "mean")
    if "t0_drawdown" in df.columns:
        agg_map["mean_t0_drawdown"] = ("t0_drawdown", "mean")

    for col in ["future_return_1d", "future_return_3d", "future_return_5d", "future_return_20d"]:
        if col in df.columns:
            agg_map[f"mean_{col}"] = (col, "mean")

    for col in ["signal_success_1d", "signal_success_3d", "signal_success_5d", "signal_success_20d"]:
        if col in df.columns:
            agg_map[f"accuracy_{col}"] = (col, "mean")

    if "regime" in df.columns:
        agg_map["strong_uptrend_rows"] = ("regime", lambda s: int((s == "strong_uptrend").sum()))
        agg_map["downtrend_rows"] = ("regime", lambda s: int((s == "downtrend").sum()))
        agg_map["high_vol_rows"] = ("regime", lambda s: int((s == "high_vol").sum()))
        agg_map["mixed_rows"] = ("regime", lambda s: int((s == "mixed").sum()))
        agg_map["unknown_rows"] = ("regime", lambda s: int((s == "unknown").sum()))
        agg_map["drawdown_stress_rows"] = ("regime", lambda s: int((s == "drawdown_stress").sum()))

    if "reason" in df.columns:
        agg_map["transition_rows"] = ("reason", lambda s: int(s.astype(str).str.startswith("regime_transition").sum()))
        agg_map["filtered_rows"] = ("reason", lambda s: int(s.astype(str).str.startswith("filtered_out_").sum()))
        agg_map["neutral_rows"] = ("reason", lambda s: int(s.astype(str).str.startswith("neutral_").sum()))

    summary = df.groupby(["seed_label"]).agg(**agg_map).reset_index()
    return summary


if __name__ == "__main__":
    df = load_timeseries_dataset()

    if df.empty:
        print("No timeseries dataset rows.")
        pd.DataFrame().to_csv(OUTPUT_SIGNALS, index=False)
    else:
        signals = []
        reasons = []
        regimes = []

        for _, row in df.iterrows():
            regime = detect_regime(row)
            signal, reason = infer_signal(row, df_history=df)

            regimes.append(regime)
            signals.append(signal)
            reasons.append(reason)

        df["regime"] = regimes
        df["signal"] = signals
        df["reason"] = reasons

        df["signal_success_1d"] = df.apply(lambda r: evaluate_signal(r, horizon=1), axis=1)
        if "future_return_3d" in df.columns:
            df["signal_success_3d"] = df.apply(lambda r: evaluate_signal(r, horizon=3), axis=1)
        if "future_return_5d" in df.columns:
            df["signal_success_5d"] = df.apply(lambda r: evaluate_signal(r, horizon=5), axis=1)
        if "future_return_20d" in df.columns:
            df["signal_success_20d"] = df.apply(lambda r: evaluate_signal(r, horizon=20), axis=1)

        summary = build_signal_summary(df)
        df.to_csv(OUTPUT_SIGNALS, index=False)

        print("Signal summary:")
        if not summary.empty:
            print(summary.to_string(index=False))
        else:
            print("No summary available.")

        print("\nRegime counts:")
        print(df["regime"].value_counts(dropna=False).to_string())

        print("\nSignal counts:")
        print(df["signal"].value_counts(dropna=False).to_string())

        print("\nReason counts:")
        print(df["reason"].astype(str).value_counts(dropna=False).head(40).to_string())

        print(f"\nSaved: {OUTPUT_SIGNALS}")