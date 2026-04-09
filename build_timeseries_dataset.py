# build_timeseries_dataset.py

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    OUTPUT_EVENT_MATCHES,
    OUTPUT_SEED_INDEX_PANEL_TS,
    OUTPUT_TIMESERIES_DATASET,
    FUTURE_HORIZON_DAYS,
)


def load_event_matches(path: str = OUTPUT_EVENT_MATCHES) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    for col in ["polymarket_time", "news_time", "event_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                utc=True,
                errors="coerce",
                format="mixed",
            )

    for col in ["match_score", "lead_lag_minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_seed_index_panel_ts(path: str = OUTPUT_SEED_INDEX_PANEL_TS) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )

    numeric_cols = [
        "close",
        "return_1d",
        "log_return_1d",
        "volatility_20d",
        "momentum_20d",
        "momentum_60d",
        "drawdown",
        "trend_50_200",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["timestamp_utc"].notna()].copy()

    sort_cols = [c for c in ["seed_label", "symbol", "timestamp_utc"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def resolve_event_time(ev: pd.Series):
    """
    Prefer event_time from matcher.
    Fall back to news_time, then polymarket_time.
    """
    event_time = ev.get("event_time")

    if pd.isna(event_time):
        event_time = ev.get("news_time")

    if pd.isna(event_time):
        event_time = ev.get("polymarket_time")

    return event_time


def last_row_on_or_before(g: pd.DataFrame, target_ts: pd.Timestamp):
    g2 = g[g["timestamp_utc"] <= target_ts]
    if g2.empty:
        return None
    return g2.sort_values("timestamp_utc").iloc[-1]


def first_row_on_or_after(g: pd.DataFrame, target_ts: pd.Timestamp):
    g2 = g[g["timestamp_utc"] >= target_ts]
    if g2.empty:
        return None
    return g2.sort_values("timestamp_utc").iloc[0]


def nth_future_row_after_index(g: pd.DataFrame, current_idx: int, n_days: int):
    target_idx = current_idx + n_days
    if target_idx >= len(g):
        return None
    return g.iloc[target_idx]


def build_timeseries_dataset() -> pd.DataFrame:
    event_matches = load_event_matches()
    seed_ts = load_seed_index_panel_ts()

    if event_matches.empty or seed_ts.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for _, ev in event_matches.iterrows():
        seed_label = ev.get("seed_label")
        event_time = resolve_event_time(ev)

        if pd.isna(seed_label) or pd.isna(event_time):
            continue

        mapped = seed_ts[seed_ts["seed_label"] == seed_label].copy()
        if mapped.empty:
            continue

        for symbol, g in mapped.groupby("symbol", sort=False):
            g = g.sort_values("timestamp_utc").reset_index(drop=True)

            symbol_latest_ts = g["timestamp_utc"].max()
            symbol_earliest_ts = g["timestamp_utc"].min()

            if pd.isna(symbol_latest_ts) or pd.isna(symbol_earliest_ts):
                continue

            if event_time < symbol_earliest_ts:
                continue

            if event_time > symbol_latest_ts:
                continue

            t0_row = last_row_on_or_before(g, event_time)
            if t0_row is None:
                continue

            t0_idx_matches = g.index[g["timestamp_utc"] == t0_row["timestamp_utc"]].tolist()
            if not t0_idx_matches:
                continue
            t0_idx = t0_idx_matches[-1]

            t0_close = t0_row.get("close")

            base = {
                "seed_label": seed_label,
                "symbol": symbol,
                "country": t0_row.get("country"),
                "index_name": t0_row.get("index_name"),
                "bucket": t0_row.get("bucket"),
                "polymarket_title": ev.get("polymarket_title"),
                "news_title": ev.get("news_title"),
                "leader": ev.get("leader"),
                "polymarket_time": ev.get("polymarket_time"),
                "news_time": ev.get("news_time"),
                "event_time": event_time,
                "match_score": ev.get("match_score"),
                "lead_lag_minutes": ev.get("lead_lag_minutes"),
                "symbol_earliest_timestamp": symbol_earliest_ts,
                "symbol_latest_timestamp": symbol_latest_ts,
                "t0_timestamp": t0_row.get("timestamp_utc"),
                "t0_close": t0_close,
                "t0_return_1d": t0_row.get("return_1d"),
                "t0_log_return_1d": t0_row.get("log_return_1d"),
                "t0_volatility_20d": t0_row.get("volatility_20d"),
                "t0_momentum_20d": t0_row.get("momentum_20d"),
                "t0_momentum_60d": t0_row.get("momentum_60d"),
                "t0_drawdown": t0_row.get("drawdown"),
                "t0_trend_50_200": t0_row.get("trend_50_200"),
            }

            for horizon in FUTURE_HORIZON_DAYS:
                target_ts = event_time + pd.Timedelta(days=horizon)
                future_row = first_row_on_or_after(g, target_ts)

                if future_row is None:
                    future_row = nth_future_row_after_index(g, t0_idx, horizon)

                has_forward = future_row is not None
                base[f"has_{horizon}d_forward"] = int(has_forward)

                if (not has_forward) or pd.isna(t0_close) or t0_close == 0:
                    base[f"t{horizon}d_timestamp"] = pd.NaT
                    base[f"t{horizon}d_close"] = np.nan
                    base[f"future_return_{horizon}d"] = np.nan
                else:
                    future_close = future_row.get("close")
                    base[f"t{horizon}d_timestamp"] = future_row.get("timestamp_utc")
                    base[f"t{horizon}d_close"] = future_close
                    base[f"future_return_{horizon}d"] = (
                        (future_close / t0_close) - 1
                        if pd.notna(future_close) and pd.notna(t0_close) and t0_close != 0
                        else np.nan
                    )

            rows.append(base)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(["seed_label", "event_time", "symbol"]).reset_index(drop=True)


if __name__ == "__main__":
    df = build_timeseries_dataset()
    df.to_csv(OUTPUT_TIMESERIES_DATASET, index=False)

    if df.empty:
        print("No timeseries rows created.")
    else:
        print(df.head(30).to_string(index=False))

        coverage_cols = [c for c in df.columns if c.startswith("has_") and c.endswith("_forward")]
        if coverage_cols:
            print("\nForward coverage:")
            for c in coverage_cols:
                print(f"{c}: {df[c].mean():.3f}")

    print(f"\nSaved: {OUTPUT_TIMESERIES_DATASET}")