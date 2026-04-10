import pandas as pd

df = pd.read_csv("data/processed/final_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

work = df.dropna(subset=["belief_mean"]).copy()

work["signal_belief"] = 0
work.loc[work["b_change"] > 0, "signal_belief"] = 1
work.loc[work["b_change"] < 0, "signal_belief"] = -1

active = work[work["signal_belief"] != 0].copy()

active["hit_1d"] = (
    ((active["signal_belief"] == 1) & (active["return_1d"] > 0)) |
    ((active["signal_belief"] == -1) & (active["return_1d"] < 0))
).astype(int)

active["hit_5d"] = (
    ((active["signal_belief"] == 1) & (active["return_5d"] > 0)) |
    ((active["signal_belief"] == -1) & (active["return_5d"] < 0))
).astype(int)

summary = pd.DataFrame([{
    "rows_with_belief": len(work),
    "active_signals": len(active),
    "long_signals": int((active["signal_belief"] == 1).sum()),
    "short_signals": int((active["signal_belief"] == -1).sum()),
    "acc_1d": active["hit_1d"].mean() if len(active) else None,
    "acc_5d": active["hit_5d"].mean() if len(active) else None,
}])

summary.to_csv("data/processed/belief_baseline_summary.csv", index=False)
print(summary.to_string(index=False))