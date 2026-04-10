import pandas as pd
from utils import save_csv, print_stage

ALLOWED_SYMBOLS = ["GLD", "^GDAXI", "USO"]
OVERLAY_FLOOR = 0.55
OVERLAY_CEILING = 0.75
COST_PER_TRADE = 0.002  # 20 bps round-trip cost


def main() -> None:
    print_stage("Backtesting filtered champion with costs")

    df = pd.read_csv("data/processed/paper_trade_overlay_log.csv")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")

    filtered = df[
        (df["direction"] == 1) &
        (df["symbol"].isin(ALLOWED_SYMBOLS)) &
        (df["overlay_score"].abs() >= OVERLAY_FLOOR) &
        (df["overlay_score"].abs() < OVERLAY_CEILING)
    ].copy()

    filtered["ret_1d_after_cost"] = filtered["ret_1d_realized"] - COST_PER_TRADE
    filtered["ret_5d_after_cost"] = filtered["ret_5d_realized"] - COST_PER_TRADE
    filtered["hit_1d_after_cost"] = (filtered["ret_1d_after_cost"] > 0).astype(int)
    filtered["hit_5d_after_cost"] = (filtered["ret_5d_after_cost"] > 0).astype(int)

    summary = pd.DataFrame([{
        "rule": "filtered_champion_after_cost",
        "n_trades": len(filtered),
        "cost_per_trade": COST_PER_TRADE,
        "mean_ret_1d_after_cost": filtered["ret_1d_after_cost"].mean(),
        "mean_ret_5d_after_cost": filtered["ret_5d_after_cost"].mean(),
        "hit_1d_after_cost": filtered["hit_1d_after_cost"].mean(),
        "hit_5d_after_cost": filtered["hit_5d_after_cost"].mean(),
    }])

    save_csv(summary, "data/processed/paper_trade_filtered_overlay_costs_summary.csv")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
