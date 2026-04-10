import pandas as pd
from utils import save_csv, print_stage


def main() -> None:
    print_stage("Building paper trade log from champion overlay")

    df = pd.read_csv("data/processed/final_dataset_with_overlay.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)

    trades = df[df["overlay_signal"] != 0].copy()

    trades["entry_date"] = trades["Date"]
    trades["entry_price"] = trades["price"]
    trades["direction"] = trades["overlay_signal"]

    trades["exit_date_1d"] = trades.groupby("symbol")["Date"].shift(-1)
    trades["exit_price_1d"] = trades.groupby("symbol")["price"].shift(-1)

    trades["exit_date_5d"] = trades.groupby("symbol")["Date"].shift(-5)
    trades["exit_price_5d"] = trades.groupby("symbol")["price"].shift(-5)

    trades["ret_1d_realized"] = (
        (trades["exit_price_1d"] - trades["entry_price"]) / trades["entry_price"]
    ) * trades["direction"]

    trades["ret_5d_realized"] = (
        (trades["exit_price_5d"] - trades["entry_price"]) / trades["entry_price"]
    ) * trades["direction"]

    trades["hit_1d_realized"] = (trades["ret_1d_realized"] > 0).astype("Int64")
    trades["hit_5d_realized"] = (trades["ret_5d_realized"] > 0).astype("Int64")

    keep_cols = [
        "symbol",
        "entry_date",
        "direction",
        "entry_price",
        "overlay_score",
        "belief_mean",
        "price_score",
        "belief_score",
        "exit_date_1d",
        "exit_price_1d",
        "ret_1d_realized",
        "hit_1d_realized",
        "exit_date_5d",
        "exit_price_5d",
        "ret_5d_realized",
        "hit_5d_realized",
    ]

    trades = trades[keep_cols].copy()

    save_csv(trades, "data/processed/paper_trade_overlay_log.csv")

    summary = pd.DataFrame([{
        "n_trades": len(trades),
        "mean_ret_1d": trades["ret_1d_realized"].mean(),
        "mean_ret_5d": trades["ret_5d_realized"].mean(),
        "hit_1d": trades["hit_1d_realized"].mean(),
        "hit_5d": trades["hit_5d_realized"].mean(),
    }])

    save_csv(summary, "data/processed/paper_trade_overlay_summary.csv")

    print(trades.head(20).to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()