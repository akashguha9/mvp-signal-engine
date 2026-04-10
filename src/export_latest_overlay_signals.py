import pandas as pd
from utils import save_csv, print_stage


def main() -> None:
    print_stage("Exporting latest overlay signals")

    df = pd.read_csv("data/processed/final_dataset_with_overlay.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()

    cols = [
        "Date",
        "symbol",
        "price",
        "momentum_20d",
        "trend_50_200",
        "drawdown",
        "belief_mean",
        "price_score",
        "belief_score",
        "overlay_score",
        "overlay_signal",
    ]

    cols = [c for c in cols if c in latest.columns]
    latest = latest[cols].sort_values("overlay_score", ascending=False).reset_index(drop=True)

    save_csv(latest, "data/processed/latest_overlay_signals.csv")

    print(f"Latest date: {latest_date.date()}")
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()