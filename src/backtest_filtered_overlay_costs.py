import pandas as pd
from utils import save_csv, print_stage

FILTERED_LOG_PATH = "data/processed/paper_trade_filtered_overlay_log.csv"
FALLBACK_INPUT_PATH = "data/processed/paper_trade_overlay_log.csv"
OUTPUT_SUMMARY_PATH = "data/processed/paper_trade_filtered_overlay_costs_summary.csv"

ALLOWED_SYMBOLS = ["GLD", "^GDAXI"]
OVERLAY_FLOOR = 0.55
OVERLAY_CEILING = 0.75
COST_PER_TRADE = 0.002


def load_filtered_trades() -> pd.DataFrame:
    try:
        df = pd.read_csv(FILTERED_LOG_PATH)
        if len(df) > 0:
            return df
    except FileNotFoundError:
        pass

    df = pd.read_csv(FALLBACK_INPUT_PATH)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")

    df = df[
        (df["direction"] == 1)
        & (df["symbol"].isin(ALLOWED_SYMBOLS))
        & (df["overlay_score"].abs() >= OVERLAY_FLOOR)
        & (df["overlay_score"].abs() < OVERLAY_CEILING)
    ].copy()

    return df


def main() -> None:
    print_stage("Backtesting filtered champion with costs")

    df = load_filtered_trades().copy()

    df["ret_1d_after_cost"] = df["ret_1d_realized"] - COST_PER_TRADE
    df["ret_5d_after_cost"] = df["ret_5d_realized"] - COST_PER_TRADE

    df["hit_1d_after_cost"] = (df["ret_1d_after_cost"] > 0).astype(int)
    df["hit_5d_after_cost"] = (df["ret_5d_after_cost"] > 0).astype(int)

    summary = pd.DataFrame(
        [
            {
                "rule": "filtered_champion_after_cost",
                "n_trades": len(df),
                "cost_per_trade": COST_PER_TRADE,
                "mean_ret_1d_after_cost": df["ret_1d_after_cost"].mean(),
                "mean_ret_5d_after_cost": df["ret_5d_after_cost"].mean(),
                "hit_1d_after_cost": df["hit_1d_after_cost"].mean(),
                "hit_5d_after_cost": df["hit_5d_after_cost"].mean(),
            }
        ]
    )

    save_csv(summary, OUTPUT_SUMMARY_PATH)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()