import pandas as pd

INPUT_PATH = "data/processed/output_timeseries_dataset.csv"
OUTPUT_PATH = "data/processed/walkforward_filtered_champion_summary.csv"

OVERLAY_FLOOR = 0.55
OVERLAY_CEILING = 0.75
ALLOWED_SYMBOLS = ["GLD", "^GDAXI"]  # ✅ UPDATED

SPLIT_DATE = "2023-01-01"

def compute_metrics(df):
    return {
        "n_trades": len(df),
        "mean_ret_1d": df["ret_1d_realized"].mean(),
        "mean_ret_5d": df["ret_5d_realized"].mean(),
        "hit_1d": (df["ret_1d_realized"] > 0).mean(),
        "hit_5d": (df["ret_5d_realized"] > 0).mean()
    }

def main():
    print("[STAGE] Walkforward filtered champion")

    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    df = df[
        (df["signal"] == 1) &
        (df["symbol"].isin(ALLOWED_SYMBOLS)) &
        (df["overlay_score"].abs() >= OVERLAY_FLOOR) &
        (df["overlay_score"].abs() < OVERLAY_CEILING)
    ].copy()

    df["ret_1d_realized"] = (df["exit_price_1d"] - df["entry_price"]) / df["entry_price"]
    df["ret_5d_realized"] = (df["exit_price_5d"] - df["entry_price"]) / df["entry_price"]

    train = df[df["Date"] < SPLIT_DATE]
    test = df[df["Date"] >= SPLIT_DATE]

    results = []

    for name, subset in [("train", train), ("test", test)]:
        metrics = compute_metrics(subset)
        metrics.update({"dataset": name, "rule": "filtered_champion"})
        results.append(metrics)

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH} | Rows: {len(results)}")


if __name__ == "__main__":
    main()