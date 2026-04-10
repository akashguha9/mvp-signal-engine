import pandas as pd
from utils import save_csv, print_stage

INPUT_PATH = "data/processed/final_dataset_with_overlay.csv"
OUTPUT_PATH = "data/processed/latest_filtered_champion_signals.csv"

ALLOWED_SYMBOLS = ["GLD", "^GDAXI"]
OVERLAY_FLOOR = 0.55
OVERLAY_CEILING = 0.75


def main() -> None:
    print_stage("Exporting latest filtered champion signals")

    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()

    latest = latest[
        (latest["symbol"].isin(ALLOWED_SYMBOLS))
        & (latest["overlay_signal"] == 1)
        & (latest["overlay_score"].abs() >= OVERLAY_FLOOR)
        & (latest["overlay_score"].abs() < OVERLAY_CEILING)
    ].copy()

    latest = latest[
        [
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
    ].sort_values(["overlay_score", "symbol"], ascending=[False, True])

    latest["Date"] = latest["Date"].dt.strftime("%Y-%m-%d")

    save_csv(latest, OUTPUT_PATH)

    print(f"Latest date: {latest_date.strftime('%Y-%m-%d')}")
    if latest.empty:
        print(latest.to_string(index=False))
    else:
        print(latest.to_string(index=False))


if __name__ == "__main__":
    main()