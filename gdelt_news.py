# gdelt_news.py

import pandas as pd
from urllib.parse import quote_plus

from helpers import safe_get_json, parse_mixed_utc


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


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
        "theme_bucket",
        "query",
        "raw",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    return df[expected_cols]


def _filter_df_by_date(df, start_date, end_date, timestamp_col="timestamp_utc"):
    if df.empty:
        return df

    df = df.copy()
    df[timestamp_col] = parse_mixed_utc(df[timestamp_col])

    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)

    df = df[df[timestamp_col].notna()].copy()
    df = df[(df[timestamp_col] >= start_ts) & (df[timestamp_col] <= end_ts)]
    return df.sort_values(timestamp_col).reset_index(drop=True)


def build_gdelt_query(base_query, theme_bucket=None):
    """
    Keep this simple and controllable.
    """
    query = base_query.strip()

    # Optional lightweight theme augmentation
    if theme_bucket == "geopolitics":
        # no forced extra terms
        return query

    if theme_bucket == "macro":
        return query

    if theme_bucket == "crypto":
        return query

    return query


def fetch_gdelt_articles(
    query,
    start_date,
    end_date,
    max_records=250,
    mode="artlist",
    theme_bucket=None,
):
    """
    GDELT DOC 2.0 article fetch.
    We use STARTDATETIME / ENDDATETIME because month-by-month backfills need explicit ranges.
    """
    gdelt_query = build_gdelt_query(query, theme_bucket=theme_bucket)

    params = {
        "query": gdelt_query,
        "mode": mode,
        "maxrecords": max_records,
        "format": "json",
        "STARTDATETIME": pd.to_datetime(start_date, utc=True).strftime("%Y%m%d%H%M%S"),
        "ENDDATETIME": pd.to_datetime(end_date, utc=True).strftime("%Y%m%d%H%M%S"),
    }

    data = safe_get_json(GDELT_DOC_API, params=params)

    # GDELT response shape can vary slightly by mode
    articles = data.get("articles", []) if isinstance(data, dict) else []

    rows = []
    for item in articles:
        rows.append({
            "asset_class": "gdelt_news",
            "source_provider": "gdelt_doc",
            "symbol_or_query": query,
            "headline": item.get("title"),
            "snippet": item.get("seendate") or item.get("domain"),
            "article_url": item.get("url"),
            "timestamp_utc": item.get("seendate") or item.get("socialimage"),
            "sentiment": None,
            "theme_bucket": theme_bucket,
            "query": gdelt_query,
            "raw": item,
        })

    df = _standardize_rows(rows)
    return _filter_df_by_date(df, start_date, end_date)


if __name__ == "__main__":
    df = fetch_gdelt_articles(
        query="Ukraine Russia NATO",
        start_date="2025-01-01",
        end_date="2025-01-31 23:59:59",
        max_records=100,
        theme_bucket="geopolitics",
    )

    print(df.head(20).to_string(index=False))
    print(f"\nRows: {len(df)}")