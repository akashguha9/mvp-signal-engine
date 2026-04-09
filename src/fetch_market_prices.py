import pandas as pd
import yfinance as yf

from config import SYMBOLS, START_DATE, END_DATE
from utils import save_csv


def fetch_market_data() -> None:
    all_data = []

    for symbol in SYMBOLS:
        print(f"Fetching market prices for {symbol}...")
        df = yf.download(symbol, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)

        if df.empty:
            print(f"Warning: no market data for {symbol}")
            continue

        df = df.reset_index()
        if "Date" not in df.columns:
            print(f"Warning: missing Date column for {symbol}")
            continue

        df["Date"] = pd.to_datetime(df["Date"]).dt.floor("D")
        df["symbol"] = symbol

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

        df["return_1d"] = df[price_col].pct_change(1)
        df["return_3d"] = df[price_col].pct_change(3)
        df["return_5d"] = df[price_col].pct_change(5)

        df["volatility_20d"] = df[price_col].pct_change().rolling(20).std()
        df["momentum_20d"] = df[price_col].pct_change(20)
        df["momentum_60d"] = df[price_col].pct_change(60)

        df["ma_50"] = df[price_col].rolling(50).mean()
        df["ma_200"] = df[price_col].rolling(200).mean()
        df["trend_50_200"] = df["ma_50"] - df["ma_200"]

        df["drawdown"] = df[price_col] / df[price_col].cummax() - 1
        df = df.rename(columns={price_col: "price"})
        all_data.append(df[[
            "Date", "symbol", "price",
            "return_1d", "return_3d", "return_5d",
            "volatility_20d", "momentum_20d", "momentum_60d",
            "ma_50", "ma_200", "trend_50_200", "drawdown"
        ]])

    if not all_data:
        raise RuntimeError("No market price data fetched.")

    final_df = pd.concat(all_data, ignore_index=True)
    save_csv(final_df, "data/processed/market_prices.csv")


if __name__ == "__main__":
    fetch_market_data()