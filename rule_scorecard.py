# rule_scorecard.py

from __future__ import annotations

import sys
import numpy as np
import pandas as pd


def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def rolling_windows_for_rule(df: pd.DataFrame, window_size: int = 300, step: int = 150) -> pd.DataFrame:
    results = []

    if df.empty or "event_time" not in df.columns:
        return pd.DataFrame(columns=["start", "end", "rows", "directional", "acc_1d", "acc_5d"])

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce", format="mixed")
    df = df[df["event_time"].notna()].sort_values("event_time").reset_index(drop=True)

    if len(df) < window_size:
        return pd.DataFrame(columns=["start", "end", "rows", "directional", "acc_1d", "acc_5d"])

    for start in range(0, len(df) - window_size + 1, step):
        end = start + window_size
        w = df.iloc[start:end].copy()

        directional = w[w["signal"].isin([1, -1])].copy()

        results.append({
            "start": start,
            "end": end,
            "rows": len(w),
            "directional": len(directional),
            "acc_1d": safe_mean(directional["signal_success_1d"]) if "signal_success_1d" in directional.columns else np.nan,
            "acc_5d": safe_mean(directional["signal_success_5d"]) if "signal_success_5d" in directional.columns else np.nan,
        })

    return pd.DataFrame(results)


def score_rule(
    rule_df: pd.DataFrame,
    window_size: int = 300,
    step: int = 150,
    min_directional: int = 50,
):
    windows = rolling_windows_for_rule(rule_df, window_size=window_size, step=step)

    if windows.empty:
        stats = {
            "rows": len(rule_df),
            "directional_rows": int(rule_df["signal"].isin([1, -1]).sum()) if "signal" in rule_df.columns else 0,
            "overall_acc_1d": safe_mean(rule_df.loc[rule_df["signal"].isin([1, -1]), "signal_success_1d"])
            if {"signal", "signal_success_1d"}.issubset(rule_df.columns) else np.nan,
            "overall_acc_5d": safe_mean(rule_df.loc[rule_df["signal"].isin([1, -1]), "signal_success_5d"])
            if {"signal", "signal_success_5d"}.issubset(rule_df.columns) else np.nan,
            "kept_windows": 0,
            "dropped_windows": 0,
            "rolling_mean_1d": np.nan,
            "rolling_std_1d": np.nan,
            "rolling_min_1d": np.nan,
            "rolling_max_1d": np.nan,
        }
        return stats, windows, windows

    kept = windows[windows["directional"] >= min_directional].copy()
    dropped = windows[windows["directional"] < min_directional].copy()

    accs = kept["acc_1d"].dropna() if "acc_1d" in kept.columns else pd.Series(dtype=float)

    stats = {
        "rows": len(rule_df),
        "directional_rows": int(rule_df["signal"].isin([1, -1]).sum()) if "signal" in rule_df.columns else 0,
        "overall_acc_1d": safe_mean(rule_df.loc[rule_df["signal"].isin([1, -1]), "signal_success_1d"])
        if {"signal", "signal_success_1d"}.issubset(rule_df.columns) else np.nan,
        "overall_acc_5d": safe_mean(rule_df.loc[rule_df["signal"].isin([1, -1]), "signal_success_5d"])
        if {"signal", "signal_success_5d"}.issubset(rule_df.columns) else np.nan,
        "kept_windows": int(len(kept)),
        "dropped_windows": int(len(dropped)),
        "rolling_mean_1d": float(accs.mean()) if len(accs) else np.nan,
        "rolling_std_1d": float(accs.std(ddof=0)) if len(accs) else np.nan,
        "rolling_min_1d": float(accs.min()) if len(accs) else np.nan,
        "rolling_max_1d": float(accs.max()) if len(accs) else np.nan,
    }

    return stats, kept, dropped


def pass_fail(row) -> str:
    directional_rows = row["directional_rows"]
    kept_windows = row["kept_windows"]
    rolling_mean = row["rolling_mean_1d"]
    rolling_std = row["rolling_std_1d"]
    rolling_min = row["rolling_min_1d"]

    if pd.isna(rolling_mean):
        return "FAIL_no_valid_windows"

    if directional_rows < 100:
        return "FAIL_too_few_directional_rows"

    if kept_windows < 3:
        return "FAIL_too_few_kept_windows"

    if rolling_mean < 0.55:
        return "FAIL_low_mean"

    if pd.notna(rolling_std) and rolling_std > 0.15:
        return "FAIL_high_std"

    if pd.notna(rolling_min) and rolling_min < 0.45:
        return "FAIL_low_min"

    return "PASS"


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "output_signals.csv"

    df = pd.read_csv(input_file)

    if df.empty:
        print("No data.")
        return

    if "event_time" not in df.columns:
        print("event_time column missing.")
        return

    if "reason" not in df.columns:
        print("reason column missing.")
        return

    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce", format="mixed")
    df = df[df["event_time"].notna()].sort_values("event_time").reset_index(drop=True)

    print("=== RULE SCORECARD ===")
    print(f"input_file: {input_file}")
    print(f"rows: {len(df)}")

    directional = df[df["signal"].isin([1, -1])].copy()

    if directional.empty:
        print("No directional rows.")
        return

    rules = sorted(directional["reason"].astype(str).dropna().unique().tolist())

    score_rows = []
    details = {}

    for rule in rules:
        sub = directional[directional["reason"].astype(str) == rule].copy()

        stats, kept, dropped = score_rule(sub)

        score_row = {
            "rule": rule,
            **stats,
        }
        score_row["verdict"] = pass_fail(score_row)

        score_rows.append(score_row)
        details[rule] = {"kept": kept, "dropped": dropped}

    scorecard = pd.DataFrame(score_rows)

    if scorecard.empty:
        print("No rule scorecard rows.")
        return

    scorecard = scorecard.sort_values(
        ["verdict", "rolling_mean_1d", "directional_rows"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    print("\n=== SCORECARD ===")
    print(scorecard.to_string(index=False))

    print("\n=== TOP CANDIDATES ===")
    top = scorecard[scorecard["verdict"].astype(str).str.startswith("PASS")].copy()

    if top.empty:
        print("No passing rules.")
    else:
        print(top.to_string(index=False))

    print("\n=== PER-RULE WINDOW DETAILS ===")
    for _, row in scorecard.iterrows():
        rule = row["rule"]
        print("\n" + "=" * 80)
        print(f"RULE: {rule}")
        print(f"VERDICT: {row['verdict']}")
        print("-" * 80)

        print("KEPT WINDOWS:")
        kept = details[rule]["kept"]
        if kept.empty:
            print("None")
        else:
            print(kept.to_string(index=False))

        print("\nDROPPED WINDOWS:")
        dropped = details[rule]["dropped"]
        if dropped.empty:
            print("None")
        else:
            print(dropped.to_string(index=False))


if __name__ == "__main__":
    main()