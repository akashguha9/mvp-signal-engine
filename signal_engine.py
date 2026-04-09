# signal_engine.py

import re
import numpy as np
import pandas as pd

from config import OUTPUT_TIMESERIES_DATASET, OUTPUT_SIGNALS, MIN_SIGNAL_MATCH_SCORE


def load_timeseries_dataset(path=OUTPUT_TIMESERIES_DATASET):
    df = pd.read_csv(path)

    datetime_cols = [
        "event_time",
        "polymarket_time",
        "news_time",
        "t0_timestamp",
        "symbol_earliest_timestamp",
        "symbol_latest_timestamp",
    ]
    datetime_cols += [c for c in df.columns if c.endswith("_timestamp")]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                utc=True,
                errors="coerce",
                format="mixed"
            )

    for col in df.columns:
        if (
            col.startswith("future_return_")
            or col.startswith("has_")
            or col in [
                "match_score",
                "lead_lag_minutes",
                "t0_close",
                "t0_return_1d",
                "t0_volatility_20d",
                "t0_momentum_20d",
                "t0_momentum_60d",
                "t0_drawdown",
                "t0_trend_50_200",
            ]
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def extract_horizons(df):
    horizons = []
    for col in df.columns:
        m = re.match(r"future_return_(\d+)d", col)
        if m:
            horizons.append(int(m.group(1)))
    return sorted(horizons)


def infer_signal(row):
    score = row.get("match_score")
    lag = row.get("lead_lag_minutes")
    vol = row.get("t0_volatility_20d")
    trend = row.get("t0_trend_50_200")
    mom20 = row.get("t0_momentum_20d")

    signal = 0
    reason = "no_signal"

    if pd.isna(score) or pd.isna(lag):
        return signal, reason

    if score >= MIN_SIGNAL_MATCH_SCORE and lag > 0:
        signal = 1
        reason = "news_led"
    elif score >= MIN_SIGNAL_MATCH_SCORE and lag < 0:
        signal = -1
        reason = "market_led"

    if pd.notna(vol) and vol > 0.04:
        reason += "_high_vol"

    if pd.notna(trend):
        if trend > 0:
            reason += "_uptrend"
        elif trend < 0:
            reason += "_downtrend"

    if pd.notna(mom20):
        if mom20 > 0:
            reason += "_mom_up"
        elif mom20 < 0:
            reason += "_mom_down"

    return signal, reason


def evaluate_signal(row, horizon):
    signal = row.get("signal", 0)
    fwd = row.get(f"future_return_{horizon}d")

    if pd.isna(fwd):
        return np.nan
    if signal == 1:
        return int(fwd > 0)
    if signal == -1:
        return int(fwd < 0)
    return np.nan


if __name__ == "__main__":
    df = load_timeseries_dataset()

    if df.empty:
        print("No timeseries dataset rows.")
        pd.DataFrame().to_csv(OUTPUT_SIGNALS, index=False)
    else:
        horizons = extract_horizons(df)

        signals = []
        reasons = []
        for _, row in df.iterrows():
            signal, reason = infer_signal(row)
            signals.append(signal)
            reasons.append(reason)

        df["signal"] = signals
        df["reason"] = reasons

        for h in horizons:
            df[f"signal_success_{h}d"] = df.apply(lambda r, hh=h: evaluate_signal(r, horizon=hh), axis=1)

        agg_dict = {
            "rows": ("signal", "count"),
            "buy_signals": ("signal", lambda s: int((s == 1).sum())),
            "sell_signals": ("signal", lambda s: int((s == -1).sum())),
            "mean_match_score": ("match_score", "mean"),
        }

        for h in horizons:
            agg_dict[f"mean_future_return_{h}d"] = (f"future_return_{h}d", "mean")
            agg_dict[f"signal_accuracy_{h}d"] = (f"signal_success_{h}d", "mean")
            agg_dict[f"coverage_{h}d"] = (f"has_{h}d_forward", "mean")

        summary = df.groupby("seed_label").agg(**agg_dict).reset_index()

        df.to_csv(OUTPUT_SIGNALS, index=False)

        print("Signal summary:")
        print(summary.to_string(index=False))
        print(f"\nSaved: {OUTPUT_SIGNALS}")