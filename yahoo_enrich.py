# yahoo_enrich.py

import pandas as pd

from config import OUTPUT_YAHOO_ENRICH
from helpers import safe_get_json
from seed_pairs import SEED_MARKETS


def fetch_yahoo_quote(symbol):
    """
    Best-effort Yahoo quote pull.
    If Yahoo blocks the request, return {} instead of crashing the pipeline.
    """
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": symbol}

    try:
        data = safe_get_json(url, params=params)
    except Exception as e:
        print(f"Yahoo quote failed for {symbol}: {e}")
        return {}

    results = ((data or {}).get("quoteResponse") or {}).get("result") or []

    if not results:
        return {}

    q = results[0]
    return {
        "symbol": symbol,
        "shortName": q.get("shortName"),
        "longName": q.get("longName"),
        "marketState": q.get("marketState"),
        "regularMarketPrice": q.get("regularMarketPrice"),
        "regularMarketChange": q.get("regularMarketChange"),
        "regularMarketChangePercent": q.get("regularMarketChangePercent"),
        "regularMarketPreviousClose": q.get("regularMarketPreviousClose"),
        "marketCap": q.get("marketCap"),
        "fiftyTwoWeekHigh": q.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": q.get("fiftyTwoWeekLow"),
        "currency": q.get("currency"),
        "quoteType": q.get("quoteType"),
        "exchange": q.get("fullExchangeName"),
        "sector_hint": q.get("financialCurrency"),
    }


def fetch_yahoo_chart(symbol, range_str="6mo", interval="1d"):
    """
    Yahoo chart endpoint often works even when quote endpoint is flaky.
    Use this as fallback enrichment.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": range_str,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits"
    }

    try:
        data = safe_get_json(url, params=params)
    except Exception as e:
        print(f"Yahoo chart failed for {symbol}: {e}")
        return pd.DataFrame()

    result = ((data or {}).get("chart") or {}).get("result") or []
    if not result:
        return pd.DataFrame()

    r = result[0]
    meta = r.get("meta") or {}
    timestamps = r.get("timestamp") or []
    quote = (((r.get("indicators") or {}).get("quote")) or [{}])[0]

    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce"),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })

    df = df[df["timestamp_utc"].notna()].copy().reset_index(drop=True)
    if df.empty:
        return df

    df["symbol"] = symbol
    df["currency"] = meta.get("currency")
    df["exchangeName"] = meta.get("exchangeName")
    df["instrumentType"] = meta.get("instrumentType")
    return df


def build_yahoo_enrichment():
    rows = []

    for seed in SEED_MARKETS:
        symbol = seed.get("yahoo_symbol")
        if not symbol:
            continue

        quote = fetch_yahoo_quote(symbol)

        if quote:
            quote["seed_label"] = seed["label"]
            quote["data_source"] = "yahoo_quote"
            rows.append(quote)
            continue

        chart_df = fetch_yahoo_chart(symbol, range_str="6mo", interval="1d")

        if not chart_df.empty:
            latest = chart_df.sort_values("timestamp_utc").iloc[-1]
            rows.append({
                "seed_label": seed["label"],
                "symbol": symbol,
                "shortName": None,
                "longName": None,
                "marketState": None,
                "regularMarketPrice": latest.get("close"),
                "regularMarketChange": None,
                "regularMarketChangePercent": None,
                "regularMarketPreviousClose": None,
                "marketCap": None,
                "fiftyTwoWeekHigh": None,
                "fiftyTwoWeekLow": None,
                "currency": latest.get("currency"),
                "quoteType": latest.get("instrumentType"),
                "exchange": latest.get("exchangeName"),
                "sector_hint": None,
                "data_source": "yahoo_chart_fallback",
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_yahoo_enrichment()

    if df.empty:
        print("No Yahoo enrichment rows.")
    else:
        print(df.to_string(index=False))
        df.to_csv(OUTPUT_YAHOO_ENRICH, index=False)
        print(f"\nSaved: {OUTPUT_YAHOO_ENRICH}")