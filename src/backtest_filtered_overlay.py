import pandas as pd
from utils import save_csv, print_stage


def main() -> None:
    print_stage("Backtesting filtered overlay")

    df = pd.read_csv("data/processed/paper_trade_overlay_log.csv")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")

    allowed_symbols = ["GLD", "^GDAXI", "USO"]

    filtered = df[
        (df["direction"] == 1) &
        (df["symbol"].isin(allowed_symbols)) &
        (df["overlay_score"].abs() >= 0.55) &
        (df["overlay_score"].abs() < 0.75)
    ].copy()

    summary = pd.DataFrame([{
        "rule": "long_only_mid_bucket_best_symbols",
        "n_trades": len(filtered),
        "symbols": ",".join(allowed_symbols),
        "mean_ret_1d": filtered["ret_1d_realized"].mean(),
        "mean_ret_5d": filtered["ret_5d_realized"].mean(),
        "hit_1d": filtered["hit_1d_realized"].mean(),
        "hit_5d": filtered["hit_5d_realized"].mean(),
    }])

    save_csv(filtered, "data/processed/paper_trade_filtered_overlay_log.csv")
    save_csv(summary, "data/processed/paper_trade_filtered_overlay_summary.csv")

    print(filtered.head(30).to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()