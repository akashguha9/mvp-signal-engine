# index_patterns.py

import numpy as np
import pandas as pd

from config import (
    OUTPUT_INDEX_HISTORY,
    OUTPUT_INDEX_FEATURES,
    OUTPUT_INDEX_FEATURES_FULL,
    OUTPUT_INDEX_CORRELATIONS,
    OUTPUT_INDEX_LEADLAG,
    INDEX_VOL_WINDOW,
    INDEX_MOMENTUM_SHORT,
    INDEX_MOMENTUM_LONG,
    INDEX_LEAD_LAG_MAX_DAYS,
)


def load_index_history(path=OUTPUT_INDEX_HISTORY):
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce",
        format="mixed"
    )
    df = df[df["timestamp_utc"].notna()].copy()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)
    return df


def add_index_features(df):
    frames = []

    for symbol, g in df.groupby("symbol", sort=False):
        g = g.copy().sort_values("timestamp_utc").reset_index(drop=True)

        g["return_1d"] = g["close"].pct_change(fill_method=None)

        ratio = g["close"] / g["close"].shift(1)
        g["log_return_1d"] = np.where(
            ratio.isna() | (ratio <= 0),
            np.nan,
            np.log(ratio)
        )

        g["volatility_20d"] = g["return_1d"].rolling(INDEX_VOL_WINDOW).std()
        g["momentum_20d"] = g["close"] / g["close"].shift(INDEX_MOMENTUM_SHORT) - 1
        g["momentum_60d"] = g["close"] / g["close"].shift(INDEX_MOMENTUM_LONG) - 1

        g["rolling_max_close"] = g["close"].cummax()
        g["drawdown"] = g["close"] / g["rolling_max_close"] - 1

        g["ma_20"] = g["close"].rolling(20).mean()
        g["ma_50"] = g["close"].rolling(50).mean()
        g["ma_200"] = g["close"].rolling(200).mean()
        g["trend_50_200"] = g["ma_50"] - g["ma_200"]

        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)


def build_latest_snapshot(features_df):
    latest = (
        features_df.sort_values(["symbol", "timestamp_utc"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def build_correlation_matrix(features_df):
    returns = features_df.pivot(index="timestamp_utc", columns="symbol", values="return_1d")

    corr = returns.corr()
    corr.index.name = "symbol_a"
    corr.columns.name = "symbol_b"

    corr_long = corr.stack().reset_index(name="correlation")
    corr_long = corr_long[corr_long["symbol_a"] != corr_long["symbol_b"]].copy()
    corr_long = corr_long.sort_values("correlation", ascending=False).reset_index(drop=True)
    return corr_long


def lagged_corr(series_a, series_b, lag):
    if lag > 0:
        return series_a.corr(series_b.shift(-lag))
    if lag < 0:
        return series_a.corr(series_b.shift(abs(lag)))
    return series_a.corr(series_b)


def build_leadlag_matrix(features_df, max_lag_days=INDEX_LEAD_LAG_MAX_DAYS):
    returns = features_df.pivot(index="timestamp_utc", columns="symbol", values="return_1d")

    symbols = list(returns.columns)
    rows = []

    for sym_a in symbols:
        for sym_b in symbols:
            if sym_a == sym_b:
                continue

            s_a = returns[sym_a]
            s_b = returns[sym_b]

            best_corr = None
            best_lag = None

            for lag in range(-max_lag_days, max_lag_days + 1):
                corr = lagged_corr(s_a, s_b, lag)
                if pd.isna(corr):
                    continue

                if best_corr is None or abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            rows.append({
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "best_lag_days": best_lag,
                "best_corr": best_corr,
                "interpretation": (
                    f"{sym_a} leads {sym_b}" if best_lag is not None and best_lag > 0
                    else f"{sym_b} leads {sym_a}" if best_lag is not None and best_lag < 0
                    else "same_day_or_unclear"
                )
            })

    out = pd.DataFrame(rows)
    out = out.sort_values("best_corr", ascending=False, key=lambda s: s.abs()).reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = load_index_history()
    features_df = add_index_features(df)
    latest_df = build_latest_snapshot(features_df)
    corr_df = build_correlation_matrix(features_df)
    leadlag_df = build_leadlag_matrix(features_df)

    features_df.to_csv(OUTPUT_INDEX_FEATURES_FULL, index=False)
    latest_df.to_csv(OUTPUT_INDEX_FEATURES, index=False)
    corr_df.to_csv(OUTPUT_INDEX_CORRELATIONS, index=False)
    leadlag_df.to_csv(OUTPUT_INDEX_LEADLAG, index=False)

    print(latest_df.head(20).to_string(index=False))
    print(f"\nSaved: {OUTPUT_INDEX_FEATURES_FULL}")
    print(f"Saved: {OUTPUT_INDEX_FEATURES}")
    print(f"Saved: {OUTPUT_INDEX_CORRELATIONS}")
    print(f"Saved: {OUTPUT_INDEX_LEADLAG}")