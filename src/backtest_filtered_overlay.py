import pandas as pd
from utils import save_csv, print_stage

INPUT_PATH = "data/processed/paper_trade_overlay_log.csv"
OUTPUT_LOG_PATH = "data/processed/paper_trade_filtered_overlay_log.csv"
OUTPUT_SUMMARY_PATH = "data/processed/paper_trade_filtered_overlay_summary.csv"

ALLOWED_SYMBOLS = ["GLD", "^GDAXI"]
OVERLAY_FLOOR = 0.55
OVERLAY_CEILING = 0.75


def main() -> None:
    print_stage("Backtesting filtered overlay")

    df = pd.read_csv(INPUT_PATH)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")

    filtered = df[
        (df["direction"] == 1)
        & (df["symbol"].isin(ALLOWED_SYMBOLS))
        & (df["overlay_score"].abs() >= OVERLAY_FLOOR)
        & (df["overlay_score"].abs() < OVERLAY_CEILING)
    ].copy()

    summary = pd.DataFrame(
        [
            {
                "rule": "long_only_mid_bucket_best_symbols_no_USO",
                "n_trades": len(filtered),
                "symbols": ",".join(ALLOWED_SYMBOLS),
                "mean_ret_1d": filtered["ret_1d_realized"].mean(),
                "mean_ret_5d": filtered["ret_5d_realized"].mean(),
                "hit_1d": filtered["hit_1d_realized"].mean(),
                "hit_5d": filtered["hit_5d_realized"].mean(),
            }
        ]
    )

    save_csv(filtered, OUTPUT_LOG_PATH)
    save_csv(summary, OUTPUT_SUMMARY_PATH)

    print(filtered.head(30).to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()