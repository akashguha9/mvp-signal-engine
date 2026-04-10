import pandas as pd

INPUT_PATH = "data/processed/output_timeseries_dataset.csv"
OUTPUT_PATH = "data/processed/latest_filtered_champion_signals.csv"

OVERLAY_FLOOR = 0.55
OVERLAY_CEILING = 0.75
ALLOWED_SYMBOLS = ["GLD", "^GDAXI"]  # ✅ UPDATED

def main():
    print("[STAGE] Exporting latest filtered champion signals")

    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    latest_date = df["Date"].max()

    latest = df[df["Date"] == latest_date]

    latest = latest[
        (latest["signal"] == 1) &
        (latest["symbol"].isin(ALLOWED_SYMBOLS)) &
        (latest["overlay_score"].abs() >= OVERLAY_FLOOR) &
        (latest["overlay_score"].abs() < OVERLAY_CEILING)
    ].copy()

    latest.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH} | Rows: {len(latest)}")
    print(f"Latest date: {latest_date}")
    print(latest)


if __name__ == "__main__":
    main()