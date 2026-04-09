# market_news.py

import pandas as pd
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

from config import FINNHUB_API_KEY
from helpers import safe_get_json, safe_get_text, to_dt, parse_mixed_utc


def _normalize_date_bounds(start_date, end_date):
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)
    return start_ts, end_ts


def _filter_df_by_date(df, start_date, end_date, timestamp_col="timestamp_utc"):
    if df.empty:
        return df

    start_ts, end_ts = _normalize_date_bounds(start_date, end_date)

    df = df.copy()
    df[timestamp_col] = parse_mixed_utc(df[timestamp_col])
    df = df[df[timestamp_col].notna()]
    df = df[(df[timestamp_col] >= start_ts) & (df[timestamp_col] <= end_ts)]
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df


def _standardize_rows(rows):
    df = pd.DataFrame(rows)

    expected_cols = [
        "asset_class",
        "source_provider",
        "symbol_or_query",
        "headline",
        "snippet",
        "article_url",
        "timestamp_utc",
        "sentiment",
        "raw",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    return df[expected_cols]


def fetch_stock_news_finnhub(symbol_or_query, start_date, end_date, api_key=None):
    api_key = api_key or FINNHUB_API_KEY
    if not api_key or api_key == "PASTE_YOUR_FINNHUB_API_KEY_HERE":
        raise ValueError("Valid Finnhub API key is required for stock_etf news.")

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol_or_query,
        "from": pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        "to": pd.to_datetime(end_date).strftime("%Y-%m-%d"),
        "token": api_key,
    }

    data = safe_get_json(url, params=params)

    rows = []
    for item in data:
        rows.append({
            "asset_class": "stock_etf",
            "source_provider": "finnhub",
            "symbol_or_query": symbol_or_query,
            "headline": item.get("headline"),
            "snippet": item.get("summary"),
            "article_url": item.get("url"),
            "timestamp_utc": to_dt(item.get("datetime")),
            "sentiment": None,
            "raw": item,
        })

    df = _standardize_rows(rows)
    return _filter_df_by_date(df, start_date, end_date)


def _parse_rss_feed(feed_url, asset_class, symbol_or_query, source_provider):
    try:
        xml_text = safe_get_text(feed_url)
        root = ET.fromstring(xml_text)
    except Exception:
        return _standardize_rows([])

    rows = []
    for item in root.findall(".//item"):
        title = item.findtext("title")
        link = item.findtext("link")
        description = item.findtext("description")
        pub_date = item.findtext("pubDate")

        rows.append({
            "asset_class": asset_class,
            "source_provider": source_provider,
            "symbol_or_query": symbol_or_query,
            "headline": title,
            "snippet": description,
            "article_url": link,
            "timestamp_utc": to_dt(pub_date),
            "sentiment": None,
            "raw": {
                "title": title,
                "link": link,
                "description": description,
                "pubDate": pub_date,
            },
        })

    return _standardize_rows(rows)


def fetch_google_news_rss(query, start_date, end_date, asset_class="geo_news"):
    q = quote_plus(query)
    feed_url = f"https://news.google.com/rss/search?q={q}"

    df = _parse_rss_feed(
        feed_url=feed_url,
        asset_class=asset_class,
        symbol_or_query=query,
        source_provider="google_news_rss",
    )
    return _filter_df_by_date(df, start_date, end_date)


def fetch_yahoo_finance_rss(query, start_date, end_date):
    q = quote_plus(query)
    feed_url = f"https://finance.yahoo.com/rss/headline?s={q}"

    df = _parse_rss_feed(
        feed_url=feed_url,
        asset_class="yahoo_finance_news",
        symbol_or_query=query,
        source_provider="yahoo_finance_rss",
    )
    return _filter_df_by_date(df, start_date, end_date)


def fetch_market_news(asset_class, api_key, symbol_or_query, start_date, end_date, extra_params=None):
    extra_params = extra_params or {}

    if asset_class == "stock_etf":
        return fetch_stock_news_finnhub(
            symbol_or_query=symbol_or_query,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

    if asset_class == "geo_news":
        return fetch_google_news_rss(
            query=symbol_or_query,
            start_date=start_date,
            end_date=end_date,
            asset_class="geo_news",
        )

    if asset_class == "yahoo_finance_news":
        return fetch_yahoo_finance_rss(
            query=symbol_or_query,
            start_date=start_date,
            end_date=end_date,
        )

    raise ValueError("asset_class must be one of: 'stock_etf', 'geo_news', 'yahoo_finance_news'")