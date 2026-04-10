import pandas as pd

INPUT_PATH = "data/processed/paper_trade_filtered_overlay_log.csv"
OUTPUT_PATH = "data/processed/paper_trade_filtered_overlay_costs_summary.csv"

COST_PER_TRADE = 0.002  # 20 bps

def main():
    print("[STAGE] Backtesting filtered champion with costs")

    df = pd.read_csv(INPUT_PATH)

    df["ret_1d_after_cost"] = df["ret_1d_realized"] - COST_PER_TRADE
    df["ret_5d_after_cost"] = df["ret_5d_realized"] - COST_PER_TRADE

    df["hit_1d_after_cost"] = (df["ret_1d_after_cost"] > 0).astype(int)
    df["hit_5d_after_cost"] = (df["ret_5d_after_cost"] > 0).astype(int)

    summary = pd.DataFrame([{
        "rule": "filtered_champion_after_cost",
        "n_trades": len(df),
        "cost_per_trade": COST_PER_TRADE,
        "mean_ret_1d_after_cost": df["ret_1d_after_cost"].mean(),
        "mean_ret_5d_after_cost": df["ret_5d_after_cost"].mean(),
        "hit_1d_after_cost": df["hit_1d_after_cost"].mean(),
        "hit_5d_after_cost": df["hit_5d_after_cost"].mean()
    }])

    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH} | Rows: {len(summary)}")


if __name__ == "__main__":
    main()