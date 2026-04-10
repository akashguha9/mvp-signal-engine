import pandas as pd

df = pd.read_csv("data/processed/final_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Keep rows where belief exists
work = df.dropna(subset=["belief_mean"]).copy()

# Very simple belief signal
work["signal_belief"] = 0
work.loc[work["b_change"] > 0, "signal_belief"] = 1
work.loc[work["b_change"] < 0, "signal_belief"] = -1

# Simple evaluation
work["hit_1d"] = (
    ((work["signal_belief"] == 1) & (work["return_1d"] > 0)) |
    ((work["signal_belief"] == -1) & (work["return_1d"] < 0))
).astype(int)

work["hit_5d"] = (
    ((work["signal_belief"] == 1) & (work["return_5d"] > 0)) |
    ((work["signal_belief"] == -1) & (work["return_5d"] < 0))
).astype(int)

summary = pd.DataFrame([{
    "rows_with_belief": len(work),
    "long_signals": int((work["signal_belief"] == 1).sum()),
    "short_signals": int((work["signal_belief"] == -1).sum()),
    "acc_1d": work.loc[work["signal_belief"] != 0, "hit_1d"].mean(),
    "acc_5d": work.loc[work["signal_belief"] != 0, "hit_5d"].mean(),
}])

summary.to_csv("data/processed/belief_baseline_summary.csv", index=False)
print(summary.to_string(index=False))