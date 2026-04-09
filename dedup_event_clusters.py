# dedup_event_clusters.py

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from config import OUTPUT_TIMESERIES_DATASET


INPUT_FILE = OUTPUT_TIMESERIES_DATASET
OUTPUT_DEDUP = "output_timeseries_dataset_dedup.csv"
OUTPUT_CLUSTER_MAP = "output_event_cluster_map.csv"


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

    numeric_cols = [
        "t0_momentum_20d",
        "t0_momentum_60d",
        "t0_trend_50_200",
        "t0_volatility_20d",
        "t0_drawdown",
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


def add_signature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def rounded(col: str, digits: int, default=np.nan):
        if col not in out.columns:
            return pd.Series([default] * len(out), index=out.index)
        return out[col].round(digits)

    out["sig_mom20"] = rounded("t0_momentum_20d", 3)
    out["sig_mom60"] = rounded("t0_momentum_60d", 3)
    out["sig_trend"] = rounded("t0_trend_50_200", 3)
    out["sig_vol"] = rounded("t0_volatility_20d", 3)
    out["sig_dd"] = rounded("t0_drawdown", 3)

    out["cluster_key"] = (
        out.get("seed_label", "").astype(str) + "||" +
        out.get("symbol", "").astype(str) + "||" +
        out["sig_mom20"].astype(str) + "||" +
        out["sig_mom60"].astype(str) + "||" +
        out["sig_trend"].astype(str) + "||" +
        out["sig_vol"].astype(str) + "||" +
        out["sig_dd"].astype(str)
    )

    return out


def assign_time_clusters(df: pd.DataFrame, max_gap_minutes: int = 180) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out = out.sort_values(["cluster_key", "event_time"], na_position="last").reset_index(drop=True)

    cluster_ids = []
    current_cluster = 0
    prev_key = None
    prev_time = None

    for _, row in out.iterrows():
        key = row["cluster_key"]
        ts = row["event_time"]

        if prev_key is None:
            current_cluster += 1
        else:
            new_cluster = False

            if key != prev_key:
                new_cluster = True
            elif pd.isna(ts) or pd.isna(prev_time):
                new_cluster = True
            else:
                gap = (ts - prev_time).total_seconds() / 60.0
                if gap > max_gap_minutes:
                    new_cluster = True

            if new_cluster:
                current_cluster += 1

        cluster_ids.append(current_cluster)
        prev_key = key
        prev_time = ts

    out["event_cluster_id"] = cluster_ids
    return out


def choose_representative(cluster_df: pd.DataFrame) -> pd.Series:
    work = cluster_df.copy()

    # Prefer rows with highest match_score, then earliest event_time
    if "match_score" in work.columns:
        work = work.sort_values(
            ["match_score", "event_time"],
            ascending=[False, True],
            na_position="last",
        )
    else:
        work = work.sort_values("event_time", ascending=True, na_position="last")

    rep = work.iloc[0].copy()

    rep["cluster_size"] = len(cluster_df)
    if "event_time" in cluster_df.columns:
        rep["cluster_start_time"] = cluster_df["event_time"].min()
        rep["cluster_end_time"] = cluster_df["event_time"].max()

    return rep


def collapse_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "event_cluster_id" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    reps = []
    maps = []

    for cluster_id, group in df.groupby("event_cluster_id", dropna=False):
        rep = choose_representative(group)
        reps.append(rep)

        for idx, row in group.iterrows():
            maps.append({
                "event_cluster_id": cluster_id,
                "original_index": idx,
                "event_time": row.get("event_time"),
                "seed_label": row.get("seed_label"),
                "symbol": row.get("symbol"),
                "cluster_key": row.get("cluster_key"),
            })

    dedup = pd.DataFrame(reps).reset_index(drop=True)
    cluster_map = pd.DataFrame(maps).reset_index(drop=True)

    return dedup, cluster_map


def main():
    print("=== DEDUP EVENT CLUSTERS ===")
    print(f"input_file: {INPUT_FILE}")

    df = load_data(INPUT_FILE)
    if df.empty:
        print("No data.")
        pd.DataFrame().to_csv(OUTPUT_DEDUP, index=False)
        pd.DataFrame().to_csv(OUTPUT_CLUSTER_MAP, index=False)
        return

    if "event_time" not in df.columns:
        print("event_time column missing.")
        return

    if "seed_label" not in df.columns or "symbol" not in df.columns:
        print("seed_label or symbol column missing.")
        return

    print(f"rows before: {len(df)}")

    df = add_signature_columns(df)
    df = assign_time_clusters(df, max_gap_minutes=180)

    dedup, cluster_map = collapse_clusters(df)

    if dedup.empty:
        print("No deduplicated output produced.")
        return

    # Keep useful cluster metadata, drop temporary sig columns if desired
    drop_cols = ["sig_mom20", "sig_mom60", "sig_trend", "sig_vol", "sig_dd"]
    keep_drop_cols = [c for c in drop_cols if c in dedup.columns]
    dedup = dedup.drop(columns=keep_drop_cols, errors="ignore")

    dedup.to_csv(OUTPUT_DEDUP, index=False)
    cluster_map.to_csv(OUTPUT_CLUSTER_MAP, index=False)

    print(f"rows after: {len(dedup)}")
    print(f"clusters: {dedup['event_cluster_id'].nunique() if 'event_cluster_id' in dedup.columns else 0}")
    if "cluster_size" in dedup.columns:
        print("cluster size summary:")
        print(dedup["cluster_size"].describe().to_string())

    print(f"saved: {OUTPUT_DEDUP}")
    print(f"saved: {OUTPUT_CLUSTER_MAP}")


if __name__ == "__main__":
    main()