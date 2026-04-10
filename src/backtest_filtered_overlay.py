import pandas as pd

INPUT_PATH = "data/processed/output_timeseries_dataset.csv"
LOG_PATH = "data/processed/paper_trade_filtered_overlay_log.csv"
SUMMARY_PATH = "data/processed/paper_trade_filtered_overlay_summary.csv"

OVERLAY_FLOOR = 0.55
OVERLAY_CEILING = 0.75
ALLOWED_SYMBOLS = ["GLD", "^GDAXI"]  # ✅ USO REMOVED

def main():
    print("[STAGE] Backtesting filtered overlay")

    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # Champion filter
    df = df[
        (df["signal"] == 1) &
        (df["symbol"].isin(ALLOWED_SYMBOLS)) &
        (df["overlay_score"].abs() >= OVERLAY_FLOOR) &
        (df["overlay_score"].abs() < OVERLAY_CEILING)
    ].copy()

    df["ret_1d_realized"] = (df["exit_price_1d"] - df["entry_price"]) / df["entry_price"]
    df["ret_5d_realized"] = (df["exit_price_5d"] - df["entry_price"]) / df["entry_price"]

    df["hit_1d_realized"] = (df["ret_1d_realized"] > 0).astype(int)
    df["hit_5d_realized"] = (df["ret_5d_realized"] > 0).astype(int)

    df.to_csv(LOG_PATH, index=False)

    summary = pd.DataFrame([{
        "rule": "long_only_mid_bucket_best_symbols",
        "n_trades": len(df),
        "symbols": ",".join(ALLOWED_SYMBOLS),
        "mean_ret_1d": df["ret_1d_realized"].mean(),
        "mean_ret_5d": df["ret_5d_realized"].mean(),
        "hit_1d": df["hit_1d_realized"].mean(),
        "hit_5d": df["hit_5d_realized"].mean()
    }])

    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Saved: {LOG_PATH} | Rows: {len(df)}")
    print(f"Saved: {SUMMARY_PATH} | Rows: {len(summary)}")


if __name__ == "__main__":
    main()