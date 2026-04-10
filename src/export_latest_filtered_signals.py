import pandas as pd
from utils import save_csv, print_stage


ALLOWED_SYMBOLS = ["GLD", "^GDAXI", "USO"]


def main() -> None:
    print_stage("Exporting latest filtered champion signals")

    df = pd.read_csv("data/processed/final_dataset_with_overlay.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    latest_date = df["Date"].max()
    latest = df[df["Date"] == latest_date].copy()

    latest = latest[
        (latest["symbol"].isin(ALLOWED_SYMBOLS)) &
        (latest["overlay_signal"] == 1) &
        (latest["overlay_score"].abs() >= 0.50) &
        (latest["overlay_score"].abs() < 0.75)
    ].copy()

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
    latest = latest[cols].sort_values("overlay_score", ascending=False)

    save_csv(latest, "data/processed/latest_filtered_champion_signals.csv")

    print(f"Latest date: {latest_date.date() if pd.notna(latest_date) else 'NA'}")
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()