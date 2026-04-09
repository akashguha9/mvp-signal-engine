# neutral_signal_probe.py

from __future__ import annotations

import sys
import numpy as np
import pandas as pd


FEATURE_COLS = [
    "t0_momentum_20d",
    "t0_momentum_60d",
    "t0_trend_50_200",
    "t0_volatility_20d",
    "t0_drawdown",
    "match_score",
    "lead_lag_minutes",
]

TARGET_COLS = [
    "future_return_1d",
    "future_return_3d",
    "future_return_5d",
]

ID_COLS = [
    "event_time",
    "seed_label",
    "symbol",
    "reason",
    "regime",
]


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe_group(df: pd.DataFrame, name: str, cols: list[str]):
    print_header(name)

    if df.empty:
        print("No rows.")
        return

    existing = [c for c in cols if c in df.columns]
    if not existing:
        print("No feature columns available.")
        return

    block = df[existing].copy()
    for c in existing:
        block[c] = pd.to_numeric(block[c], errors="coerce")

    print(block.describe().to_string())


def top_rows(df: pd.DataFrame, sort_col: str, n: int = 30):
    if df.empty or sort_col not in df.columns:
        print("No rows.")
        return

    show_cols = [c for c in ID_COLS + FEATURE_COLS + TARGET_COLS if c in df.columns]
    out = df.copy()
    for c in FEATURE_COLS + TARGET_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values(sort_col, ascending=False)
    print(out[show_cols].head(n).to_string(index=False))


def compare_means(pos: pd.DataFrame, nonpos: pd.DataFrame, cols: list[str]):
    rows = []
    for c in cols:
        if c not in pos.columns or c not in nonpos.columns:
            continue

        pos_vals = pd.to_numeric(pos[c], errors="coerce").dropna()
        non_vals = pd.to_numeric(nonpos[c], errors="coerce").dropna()

        rows.append({
            "feature": c,
            "positive_mean": float(pos_vals.mean()) if len(pos_vals) else np.nan,
            "non_positive_mean": float(non_vals.mean()) if len(non_vals) else np.nan,
            "delta": (
                float(pos_vals.mean()) - float(non_vals.mean())
                if len(pos_vals) and len(non_vals)
                else np.nan
            ),
            "positive_count": int(len(pos_vals)),
            "non_positive_count": int(len(non_vals)),
        })

    cmp_df = pd.DataFrame(rows)
    if cmp_df.empty:
        print("No comparable features.")
        return

    print(cmp_df.sort_values("delta", ascending=False).to_string(index=False))


def bucket_report(df: pd.DataFrame, feature: str, target: str = "future_return_1d", q: int = 5):
    print_header(f"BUCKET REPORT: {feature} vs {target}")

    if feature not in df.columns or target not in df.columns:
        print("Required columns missing.")
        return

    work = df[[feature, target]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna()

    if work.empty:
        print("No usable rows.")
        return

    if work[feature].nunique() < q:
        print("Not enough unique values for buckets.")
        return

    work["bucket"] = pd.qcut(work[feature], q=q, duplicates="drop")

    out = (
        work.groupby("bucket")
        .agg(
            rows=(target, "count"),
            avg_target=(target, "mean"),
            hit_rate=(target, lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )

    print(out.to_string(index=False))


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "output_signals.csv"

    df = pd.read_csv(input_file)

    if df.empty:
        print("No data.")
        return

    if "reason" not in df.columns:
        print("reason column missing.")
        return

    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce", format="mixed")

    for c in FEATURE_COLS + TARGET_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    neutral = df[df["reason"].astype(str).str.startswith("neutral_")].copy()

    print_header("NEUTRAL SIGNAL PROBE")
    print(f"input_file: {input_file}")
    print(f"total rows: {len(df)}")
    print(f"neutral rows: {len(neutral)}")

    if neutral.empty:
        print("No neutral rows found.")
        return

    # Use future_return_1d as primary split
    if "future_return_1d" not in neutral.columns:
        print("future_return_1d column missing.")
        return

    neutral_pos = neutral[neutral["future_return_1d"] > 0].copy()
    neutral_nonpos = neutral[neutral["future_return_1d"] <= 0].copy()

    print_header("NEUTRAL ROW SPLIT")
    print(f"neutral positive 1d rows: {len(neutral_pos)}")
    print(f"neutral non-positive 1d rows: {len(neutral_nonpos)}")

    describe_group(neutral_pos, "NEUTRAL POSITIVE 1D — FEATURE DESCRIBE", FEATURE_COLS)
    describe_group(neutral_nonpos, "NEUTRAL NON-POSITIVE 1D — FEATURE DESCRIBE", FEATURE_COLS)

    print_header("FEATURE MEAN COMPARISON")
    compare_means(neutral_pos, neutral_nonpos, FEATURE_COLS)

    print_header("TOP 30 NEUTRAL ROWS BY future_return_1d")
    top_rows(neutral, "future_return_1d", n=30)

    print_header("TOP 30 NEUTRAL ROWS BY future_return_5d")
    if "future_return_5d" in neutral.columns:
        top_rows(neutral, "future_return_5d", n=30)
    else:
        print("future_return_5d not available.")

    # Bucket reports for the main features
    for feat in ["t0_momentum_20d", "t0_trend_50_200", "t0_volatility_20d", "t0_drawdown"]:
        if feat in neutral.columns:
            bucket_report(neutral, feat, target="future_return_1d", q=5)

    # Reason-level neutral diagnostics
    print_header("NEUTRAL REASON COUNTS")
    print(neutral["reason"].astype(str).value_counts().to_string())

    if "regime" in neutral.columns:
        print_header("NEUTRAL REGIME COUNTS")
        print(neutral["regime"].astype(str).value_counts().to_string())

    # Show neutral rows that would have been "good misses"
    print_header("GOOD MISSES (neutral rows with strong future_return_1d)")
    good_miss = neutral[neutral["future_return_1d"] > neutral["future_return_1d"].quantile(0.90)].copy()
    if good_miss.empty:
        print("No good misses.")
    else:
        show_cols = [c for c in ID_COLS + FEATURE_COLS + TARGET_COLS if c in good_miss.columns]
        print(good_miss.sort_values("future_return_1d", ascending=False)[show_cols].head(50).to_string(index=False))


if __name__ == "__main__":
    main()