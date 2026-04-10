import pandas as pd
from utils import save_csv, print_stage


def build_final_dataset() -> None:
    print_stage("Building final dataset (Polymarket-only mode, shifted belief dates)")

    prices = pd.read_csv("data/processed/market_prices.csv")
    poly = pd.read_csv("data/processed/polymarket_prices_daily.csv")

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.floor("D")
    poly["Date"] = pd.to_datetime(poly["Date"], errors="coerce").dt.floor("D")

    prices = prices.dropna(subset=["Date"]).copy()
    poly = poly.dropna(subset=["Date"]).copy()

    # shift belief dates back by 1 day to align with lagging market-price coverage
    poly["Date"] = poly["Date"] - pd.Timedelta(days=1)

    poly_daily = (
        poly.groupby("Date", as_index=False)["polymarket_prob"]
        .mean()
        .rename(columns={"polymarket_prob": "belief_mean"})
    )

    merged = prices.merge(poly_daily, on="Date", how="left")

    merged = merged.sort_values(["symbol", "Date"]).reset_index(drop=True)
    merged["belief_mean"] = merged.groupby("symbol")["belief_mean"].ffill()

    merged["b_change"] = merged.groupby("symbol")["belief_mean"].diff(1)
    merged["b_momentum_3"] = merged.groupby("symbol")["belief_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    merged["b_volatility_3"] = merged.groupby("symbol")["belief_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).std()
    )

    save_csv(merged, "data/processed/final_dataset.csv")
    print(f"Final dataset rows: {len(merged)}")
    print("Columns:", list(merged.columns))


if __name__ == "__main__":
    build_final_dataset()