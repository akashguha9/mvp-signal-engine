import pandas as pd
import yfinance as yf

from config import SYMBOLS, START_DATE, END_DATE
from utils import save_csv, print_stage


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in df.columns.to_flat_index()
        ]
    return df


def _find_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if str(col).lower() in {"date", "datetime"}:
            return col
    return None


def _find_price_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Adj Close", "Close",
        "Adj Close_SPY", "Close_SPY",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {str(c).lower(): c for c in df.columns}
    for key in ["adj close", "close"]:
        if key in lower_map:
            return lower_map[key]

    for col in df.columns:
        c = str(col).lower()
        if "adj close" in c:
            return col
    for col in df.columns:
        c = str(col).lower()
        if c.endswith("close") or "close_" in c:
            return col
    return None


def fetch_market_data() -> None:
    print_stage("Fetching market prices")
    all_data = []

    for symbol in SYMBOLS:
        print(f"Fetching market prices for {symbol}...")

        try:
            df = yf.download(
                symbol,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=False,
            )
        except Exception as e:
            print(f"Market fetch failed for {symbol}: {e}")
            continue

        if df.empty:
            print(f"Warning: no market data for {symbol}")
            continue

        df = _flatten_columns(df).reset_index()

        date_col = _find_date_column(df)
        if date_col is None:
            print(f"Warning: missing Date column for {symbol}. Columns: {list(df.columns)}")
            continue

        price_col = _find_price_column(df)
        if price_col is None:
            print(f"Warning: missing Close/Adj Close column for {symbol}. Columns: {list(df.columns)}")
            continue

        df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
        df["symbol"] = symbol
        df["price"] = pd.to_numeric(df[price_col], errors="coerce")

        df = df.dropna(subset=["Date", "price"]).sort_values("Date").copy()

        df["return_1d"] = df["price"].pct_change(1)
        df["return_3d"] = df["price"].pct_change(3)
        df["return_5d"] = df["price"].pct_change(5)

        df["volatility_20d"] = df["price"].pct_change().rolling(20).std()
        df["momentum_20d"] = df["price"].pct_change(20)
        df["momentum_60d"] = df["price"].pct_change(60)

        df["ma_50"] = df["price"].rolling(50).mean()
        df["ma_200"] = df["price"].rolling(200).mean()
        df["trend_50_200"] = df["ma_50"] - df["ma_200"]

        df["drawdown"] = df["price"] / df["price"].cummax() - 1

        out = df[[
            "Date", "symbol", "price",
            "return_1d", "return_3d", "return_5d",
            "volatility_20d", "momentum_20d", "momentum_60d",
            "ma_50", "ma_200", "trend_50_200", "drawdown"
        ]].copy()

        print(f"Fetched {symbol}: {len(out)} rows")
        all_data.append(out)

    if not all_data:
        raise RuntimeError("No market price data fetched.")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(["symbol", "Date"]).drop_duplicates(["symbol", "Date"])
    save_csv(final_df, "data/processed/market_prices.csv")