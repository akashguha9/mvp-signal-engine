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


def map_symbol_theme(symbol: str) -> str:
    if symbol in ["USO", "GLD", "^VIX", "^GDAXI"]:
        return "belief_geopolitics"
    if symbol in ["QQQ", "SPY"]:
        return "belief_elections"
    return "belief_other"


def add_belief_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)

    df["b_change"] = df.groupby("symbol")["belief_mean"].diff(1)

    df["b_momentum_3"] = df.groupby("symbol")["belief_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    df["b_volatility_3"] = df.groupby("symbol")["belief_mean"].transform(
        lambda s: s.rolling(3, min_periods=1).std()
    )

    df["b_zscore_3"] = df.groupby("symbol")["belief_mean"].transform(
        lambda s: (
            (s - s.rolling(3, min_periods=1).mean()) /
            s.rolling(3, min_periods=1).std().replace(0, pd.NA)
        )
    )

    return df


def build_final_dataset() -> None:
    print_stage("Building final dataset (Polymarket-only mode, themed + shifted belief dates)")

    prices = pd.read_csv("data/processed/market_prices.csv")
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.floor("D")
    prices = prices.dropna(subset=["Date"]).copy()

    poly = pd.read_csv("data/processed/polymarket_prices_daily.csv")
    poly["Date"] = pd.to_datetime(poly["Date"], errors="coerce").dt.floor("D")
    poly = poly.dropna(subset=["Date"]).copy()

    # Guard: if Polymarket file is empty, still build a stable dataset with empty belief columns
    if poly.empty:
        merged = prices.copy()
        merged["belief_crypto"] = pd.NA
        merged["belief_elections"] = pd.NA
        merged["belief_geopolitics"] = pd.NA
        merged["belief_master_col"] = merged["symbol"].map(map_symbol_theme)
        merged["belief_mean"] = pd.NA

        merged = add_belief_features(merged)

        save_csv(merged, "data/processed/final_dataset.csv")
        print(f"Final dataset rows: {len(merged)}")
        print("Columns:", list(merged.columns))
        return

    markets = pd.read_csv("data/processed/polymarket_markets.csv")
    markets = markets[["market_id", "question"]].copy()

    # Join market question text
    poly = poly.merge(markets, on="market_id", how="left")

    # Theme classification
    poly["theme"] = poly["question"].apply(classify_market)

    # Shift belief dates back by 1 day
    poly["Date"] = poly["Date"] - pd.Timedelta(days=1)

    # Build themed daily belief panel
    poly_daily = (
        poly.groupby(["Date", "theme"], as_index=False)["polymarket_prob"]
        .mean()
        .pivot(index="Date", columns="theme", values="polymarket_prob")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # Ensure expected themed columns always exist
    for col in ["crypto", "elections", "geopolitics"]:
        if col not in poly_daily.columns:
            poly_daily[col] = pd.NA

    poly_daily = poly_daily.rename(columns={
        "crypto": "belief_crypto",
        "elections": "belief_elections",
        "geopolitics": "belief_geopolitics",
    })

    keep_cols = ["Date", "belief_crypto", "belief_elections", "belief_geopolitics"]
    poly_daily = poly_daily[keep_cols].copy()

    # Merge into price panel
    merged = prices.merge(poly_daily, on="Date", how="left")
    merged = merged.sort_values(["symbol", "Date"]).reset_index(drop=True)

    # Forward-fill themed beliefs within each symbol
    for col in ["belief_crypto", "belief_elections", "belief_geopolitics"]:
        merged[col] = merged.groupby("symbol")[col].ffill()

    # Map asset -> belief channel
    merged["belief_master_col"] = merged["symbol"].map(map_symbol_theme)
    merged["belief_mean"] = pd.NA

    for belief_col in ["belief_crypto", "belief_elections", "belief_geopolitics"]:
        mask = merged["belief_master_col"] == belief_col
        merged.loc[mask, "belief_mean"] = merged.loc[mask, belief_col]

    merged["belief_mean"] = pd.to_numeric(merged["belief_mean"], errors="coerce")

    merged = add_belief_features(merged)

    save_csv(merged, "data/processed/final_dataset.csv")
    print(f"Final dataset rows: {len(merged)}")
    print("Columns:", list(merged.columns))


if __name__ == "__main__":
    build_final_dataset()