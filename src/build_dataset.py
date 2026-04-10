import pandas as pd
from utils import save_csv, print_stage


def classify_market(question: str) -> str:
    q = str(question).lower()

    if any(x in q for x in ["eth", "bitcoin", "btc", "ethereum", "crypto", "airdrop", "fdv"]):
        return "crypto"

    if any(x in q for x in ["xi", "putin", "russia", "ukraine", "iran", "israel", "china", "taiwan"]):
        return "geopolitics"

    if any(x in q for x in ["election", "presidential", "president", "democratic nomination"]):
        return "elections"

    return "other"


def build_final_dataset() -> None:
    print_stage("Building final dataset (Polymarket-only mode, themed + shifted belief dates)")

    prices = pd.read_csv("data/processed/market_prices.csv")
    poly = pd.read_csv("data/processed/polymarket_prices_daily.csv")
    markets = pd.read_csv("data/processed/polymarket_markets.csv")

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.floor("D")
    poly["Date"] = pd.to_datetime(poly["Date"], errors="coerce").dt.floor("D")

    prices = prices.dropna(subset=["Date"]).copy()
    poly = poly.dropna(subset=["Date"]).copy()

    # join question text into poly rows
    markets = markets[["market_id", "question"]].copy()
    poly = poly.merge(markets, on="market_id", how="left")

    # theme classification
    poly["theme"] = poly["question"].apply(classify_market)

    # shift belief dates back by 1 day
    poly["Date"] = poly["Date"] - pd.Timedelta(days=1)

    # one daily belief series per theme
    poly_daily = (
        poly.groupby(["Date", "theme"], as_index=False)["polymarket_prob"]
        .mean()
        .pivot(index="Date", columns="theme", values="polymarket_prob")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # rename columns to consistent feature names
    rename_map = {}
    for c in poly_daily.columns:
        if c != "Date":
            rename_map[c] = f"belief_{c}"
    poly_daily = poly_daily.rename(columns=rename_map)

    merged = prices.merge(poly_daily, on="Date", how="left")
    merged = merged.sort_values(["symbol", "Date"]).reset_index(drop=True)

    belief_cols = [c for c in merged.columns if c.startswith("belief_")]

    # forward fill belief columns within each symbol
    for col in belief_cols:
        merged[col] = merged.groupby("symbol")[col].ffill()

    # symbol-specific master belief
    def map_symbol_theme(symbol: str) -> str:
        if symbol in ["USO", "GLD", "^VIX", "^GDAXI"]:
            return "belief_geopolitics"
        if symbol in ["QQQ", "SPY"]:
            return "belief_elections"
        return "belief_other"

    merged["belief_master_col"] = merged["symbol"].map(map_symbol_theme)
    merged["belief_mean"] = pd.NA

    for belief_col in [c for c in belief_cols if c in merged.columns]:
        mask = merged["belief_master_col"] == belief_col
        merged.loc[mask, "belief_mean"] = merged.loc[mask, belief_col]

    merged["belief_mean"] = pd.to_numeric(merged["belief_mean"], errors="coerce")

    # belief features
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