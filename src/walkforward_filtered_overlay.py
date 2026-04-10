import pandas as pd
from utils import save_csv, print_stage


ALLOWED_SYMBOLS = ["GLD", "^GDAXI", "USO"]


def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "dataset": label,
        "rule": "filtered_champion",
        "n_trades": len(df),
        "mean_ret_1d": df["ret_1d_realized"].mean() if len(df) else None,
        "mean_ret_5d": df["ret_5d_realized"].mean() if len(df) else None,
        "hit_1d": df["hit_1d_realized"].mean() if len(df) else None,
        "hit_5d": df["hit_5d_realized"].mean() if len(df) else None,
    }


def main() -> None:
    print_stage("Walkforward filtered champion")

    df = pd.read_csv("data/processed/paper_trade_overlay_log.csv")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")

    df = df[
        (df["direction"] == 1) &
        (df["symbol"].isin(ALLOWED_SYMBOLS)) &
        (df["overlay_score"].abs() >= 0.50) &
        (df["overlay_score"].abs() < 0.75)
    ].copy()

    split_date = df["entry_date"].quantile(0.7)

    train = df[df["entry_date"] <= split_date].copy()
    test = df[df["entry_date"] > split_date].copy()

    summary = pd.DataFrame([
        summarize(train, "train"),
        summarize(test, "test"),
    ])

    save_csv(summary, "data/processed/walkforward_filtered_champion_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()