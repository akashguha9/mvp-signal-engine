import pandas as pd

df = pd.read_csv("data/processed/final_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

work = df.dropna(subset=["belief_mean"]).copy()

# Rule A: belief vs rolling mean
work["signal_a"] = 0
work.loc[work["belief_mean"] >= work["b_momentum_3"], "signal_a"] = 1
work.loc[work["belief_mean"] < work["b_momentum_3"], "signal_a"] = -1

# Rule B: daily belief change
work["signal_b"] = 0
work.loc[work["b_change"] > 0, "signal_b"] = 1
work.loc[work["b_change"] < 0, "signal_b"] = -1

# Rule C: belief z-score
work["signal_c"] = 0
work.loc[work["b_zscore_3"] > 0, "signal_c"] = 1
work.loc[work["b_zscore_3"] < 0, "signal_c"] = -1


def score(active: pd.DataFrame, signal_col: str):
    active = active[active[signal_col] != 0].copy()
    if len(active) == 0:
        return {
            "active_signals": 0,
            "long_signals": 0,
            "short_signals": 0,
            "acc_1d": None,
            "acc_5d": None,
        }

    active["hit_1d"] = (
        ((active[signal_col] == 1) & (active["return_1d"] > 0)) |
        ((active[signal_col] == -1) & (active["return_1d"] < 0))
    ).astype(int)

    active["hit_5d"] = (
        ((active[signal_col] == 1) & (active["return_5d"] > 0)) |
        ((active[signal_col] == -1) & (active["return_5d"] < 0))
    ).astype(int)

    return {
        "active_signals": len(active),
        "long_signals": int((active[signal_col] == 1).sum()),
        "short_signals": int((active[signal_col] == -1).sum()),
        "acc_1d": active["hit_1d"].mean(),
        "acc_5d": active["hit_5d"].mean(),
    }


summary = pd.DataFrame([
    {"rule": "belief_vs_momentum3", "rows_with_belief": len(work), **score(work, "signal_a")},
    {"rule": "belief_change", "rows_with_belief": len(work), **score(work, "signal_b")},
    {"rule": "belief_zscore_3", "rows_with_belief": len(work), **score(work, "signal_c")},
])

summary.to_csv("data/processed/belief_baseline_summary.csv", index=False)
print(summary.to_string(index=False))