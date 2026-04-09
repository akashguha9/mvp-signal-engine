# seed_pairs.py

"""
Central definition of all seeds used across the pipeline.

Each seed drives:
- News ingestion (GDELT / Finnhub / etc.)
- Prediction market matching (Polymarket / Kalshi)
- Yahoo enrichment
- Index mapping layer
"""

SEED_MARKETS = [

    # ================================
    # MACRO (US / Global)
    # ================================
    {
        "label": "macro_spy",
        "asset_class": "stock_etf",
        "news_symbol_or_query": "SPY",
        "news_extra_params": {},
        "polymarket_keywords": [
            "inflation",
            "fed",
            "interest rates",
            "recession",
            "gdp",
            "jobs",
            "energy prices",
        ],
        "kalshi_tickers": [],
        "yahoo_symbol": "SPY",
        "theme_bucket": "macro",
        "historical_start": "2020-01-01",
    },

    # ================================
    # GEOPOLITICS (Ukraine / Russia / NATO)
    # ================================
    {
        "label": "ukraine_current",
        "asset_class": "geopolitics",
        "news_symbol_or_query": "ukraine russia nato war",
        "news_extra_params": {},
        "polymarket_keywords": [
            "ukraine",
            "russia",
            "nato",
            "troops",
            "invade",
            "sovereignty",
            "election",
        ],
        "kalshi_tickers": [],
        "yahoo_symbol": None,
        "theme_bucket": "geopolitics",
        "historical_start": "2020-01-01",
    },

    # ================================
    # GLOBAL INDEX LAYER (NEW)
    # ================================
    {
        "label": "global_indexes",
        "asset_class": "country_index",
        "news_symbol_or_query": None,
        "news_extra_params": {},
        "polymarket_keywords": [],
        "kalshi_tickers": [],
        "yahoo_symbol": None,
        "theme_bucket": "global_macro",
        "historical_start": "2020-01-01",
    },

]


def get_seeds():
    """Return all seeds"""
    return SEED_MARKETS


def get_seed_by_label(label):
    """Fetch a single seed by label"""
    for s in SEED_MARKETS:
        if s["label"] == label:
            return s
    return None