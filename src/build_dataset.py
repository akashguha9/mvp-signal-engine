import pandas as pd
from utils import save_csv, print_stage


def build_final_dataset() -> None:
    print_stage("Building final dataset (Polymarket-only mode, shifted belief dates)")

    prices = pd.read_csv("data/processed/market_prices.csv")
    poly = pd.read_csv("data/processed/polymarket_prices_daily.csv")

    # Parse and normalize dates
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.floor("D")
    poly["Date"] = pd.to_datetime(poly["Date"], errors="coerce").dt.floor("D")

    prices = prices.dropna(subset=["Date"]).copy()
    poly = poly.dropna(subset=["Date"]).copy()

    # Shift belief dates back by 1 day so 2026-04-09 belief can align to 2026-04-08 prices
    poly["Date"] = poly["Date"] - pd.Timedelta(days=1)

    # Collapse to one daily belief series across fetched Polymarket markets
    poly_daily = (
        poly.groupby("Date", as_index=False)["polymarket_prob"]
        .mean()
        .rename(columns={"polymarket_prob": "belief_mean"})
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Plain left merge on Date
    merged = prices.merge(poly_daily, on="Date", how="left")

    # Sort for feature generation
    merged = merged.sort_values(["symbol", "Date"]).reset_index(drop=True)

    # Forward-fill belief inside each symbol
    merged["belief_mean"] = merged.groupby("symbol")["belief_mean"].ffill()

    # Belief features
    merged["b_change"] = merged.groupby("symbol")["belief_mean"].diff(1)

    merged["b_momentum_3"] = merged.groupby("symbol")["belief_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    merged["b_volatility_3"] = merged.groupby("symbol")["belief_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).std()
    )

    merged["b_zscore_3"] = merged.groupby("symbol")["belief_mean"].transform(
        lambda s: (
            (s - s.rolling(3, min_periods=1).mean()) /
            s.rolling(3, min_periods=1).std().replace(0, pd.NA)
        )
    )

    save_csv(merged, "data/processed/final_dataset.csv")
    print(f"Final dataset rows: {len(merged)}")
    print("Columns:", list(merged.columns))


if __name__ == "__main__":
    build_final_dataset()