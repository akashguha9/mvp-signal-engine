# backtest_signals.py

import pandas as pd

SIGNAL_FILE = "output_signals.csv"
OUT_FILE = "output_backtest_summary.csv"


df = pd.read_csv(SIGNAL_FILE)

numeric_cols = [
    "signal",
    "match_score",
    "future_return_1d",
    "future_return_3d",
    "future_return_5d",
    "signal_success_1d",
    "signal_success_3d",
    "signal_success_5d",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Simple strategy PnL:
# long signal=1 gets +future return
# short signal=-1 gets -future return
for horizon in [1, 3, 5]:
    fwd_col = f"future_return_{horizon}d"
    pnl_col = f"strategy_pnl_{horizon}d"

    if fwd_col in df.columns:
        df[pnl_col] = 0.0
        df.loc[df["signal"] == 1, pnl_col] = df.loc[df["signal"] == 1, fwd_col]
        df.loc[df["signal"] == -1, pnl_col] = -df.loc[df["signal"] == -1, fwd_col]

summary_rows = []

for seed_label, g in df.groupby("seed_label", dropna=False):
    row = {
        "seed_label": seed_label,
        "rows": len(g),
        "buy_signals": int((g["signal"] == 1).sum()) if "signal" in g.columns else 0,
        "sell_signals": int((g["signal"] == -1).sum()) if "signal" in g.columns else 0,
        "avg_match_score": g["match_score"].mean() if "match_score" in g.columns else None,
    }

    for horizon in [1, 3, 5]:
        pnl_col = f"strategy_pnl_{horizon}d"
        succ_col = f"signal_success_{horizon}d"

        if pnl_col in g.columns:
            row[f"mean_strategy_pnl_{horizon}d"] = g[pnl_col].mean()
            row[f"sum_strategy_pnl_{horizon}d"] = g[pnl_col].sum()

        if succ_col in g.columns:
            row[f"signal_accuracy_{horizon}d"] = g[succ_col].mean()

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_FILE, index=False)

print(summary.to_string(index=False))
print(f"\nSaved: {OUT_FILE}")