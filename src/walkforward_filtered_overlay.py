import pandas as pd
from utils import save_csv, print_stage

INPUT_PATH = "data/processed/paper_trade_overlay_log.csv"
OUTPUT_SUMMARY_PATH = "data/processed/walkforward_filtered_champion_summary.csv"

ALLOWED_SYMBOLS = ["GLD", "^GDAXI"]
OVERLAY_FLOOR = 0.55
OVERLAY_CEILING = 0.75
TRAIN_RATIO = 0.70


def summarize_block(df: pd.DataFrame, dataset_name: str) -> dict:
    return {
        "dataset": dataset_name,
        "rule": "filtered_champion",
        "n_trades": len(df),
        "mean_ret_1d": df["ret_1d_realized"].mean(),
        "mean_ret_5d": df["ret_5d_realized"].mean(),
        "hit_1d": df["hit_1d_realized"].mean(),
        "hit_5d": df["hit_5d_realized"].mean(),
    }


def main() -> None:
    print_stage("Walkforward filtered champion")

    df = pd.read_csv(INPUT_PATH)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df = df.sort_values("entry_date").reset_index(drop=True)

    filtered = df[
        (df["direction"] == 1)
        & (df["symbol"].isin(ALLOWED_SYMBOLS))
        & (df["overlay_score"].abs() >= OVERLAY_FLOOR)
        & (df["overlay_score"].abs() < OVERLAY_CEILING)
    ].copy()

    split_idx = int(len(filtered) * TRAIN_RATIO)
    train_df = filtered.iloc[:split_idx].copy()
    test_df = filtered.iloc[split_idx:].copy()

    summary = pd.DataFrame(
        [
            summarize_block(train_df, "train"),
            summarize_block(test_df, "test"),
        ]
    )

    save_csv(summary, OUTPUT_SUMMARY_PATH)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()