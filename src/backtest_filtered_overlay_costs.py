import pandas as pd
from utils import save_csv, print_stage


COST_PER_TRADE = 0.002  # 0.20% round-trip
ALLOWED_SYMBOLS = ["GLD", "^GDAXI", "USO"]


def main() -> None:
    print_stage("Backtesting filtered champion with costs")

    df = pd.read_csv("data/processed/paper_trade_overlay_log.csv")

    df = df[
        (df["direction"] == 1) &
        (df["symbol"].isin(ALLOWED_SYMBOLS)) &
        (df["overlay_score"].abs() >= 0.50) &
        (df["overlay_score"].abs() < 0.75)
    ].copy()

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
        "hit_5d_after_cost": df["hit_5d_after_cost"].mean(),
    }])

    save_csv(summary, "data/processed/paper_trade_filtered_overlay_costs_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()