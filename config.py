# config.py

import os

# ============================================
# API KEYS
# ============================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d7b971pr01qlbg01e9t0d7b971pr01qlbg01e9tg")
GDELT_API_KEY = os.getenv("GDELT_API_KEY", "")

# ============================================
# BASE URLS
# ============================================

POLY_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLY_CLOB_BASE = "https://clob.polymarket.com"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# ============================================
# GLOBAL DATE RANGE
# ============================================

GLOBAL_START_DATE = "2020-01-01"
DEFAULT_SIGNAL_START_DATE = GLOBAL_START_DATE

# ============================================
# REQUEST / HTTP SETTINGS
# ============================================

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

REQUEST_SLEEP_SECONDS = 1.0
MAX_RECORDS_PER_CALL = 150
RETRY_ATTEMPTS = 5

# ============================================
# CORE OUTPUT FILES
# ============================================

OUTPUT_SEED_SUMMARY = "output_seed_summary.csv"
OUTPUT_PRICE_PANEL = "output_price_panel.csv"
OUTPUT_NEWS_PANEL = "output_news_panel.csv"
OUTPUT_EVENT_PANEL = "output_event_panel.csv"
OUTPUT_EVENT_MATCHES = "output_event_matches.csv"
OUTPUT_YAHOO_ENRICH = "output_yahoo_enrich.csv"
OUTPUT_MODEL_DATASET = "output_model_dataset.csv"

# ============================================
# INDEX LAYER OUTPUTS
# ============================================

OUTPUT_INDEX_HISTORY = "output_index_history.csv"
OUTPUT_INDEX_FEATURES = "output_index_features.csv"              # latest snapshot per symbol
OUTPUT_INDEX_FEATURES_FULL = "output_index_features_full.csv"    # full timeseries features
OUTPUT_INDEX_CORRELATIONS = "output_index_correlations.csv"
OUTPUT_INDEX_LEADLAG = "output_index_leadlag.csv"
OUTPUT_INDEX_MAPPING = "output_index_mapping.csv"
OUTPUT_SEED_INDEX_PANEL = "output_seed_index_panel.csv"          # latest snapshot joined to seeds
OUTPUT_SEED_INDEX_PANEL_TS = "output_seed_index_panel_ts.csv"    # full timeseries joined to seeds

# ============================================
# TIMESERIES / SIGNAL OUTPUTS
# ============================================

OUTPUT_TIMESERIES_DATASET = "output_timeseries_dataset.csv"
OUTPUT_SIGNALS = "output_signals.csv"

# ============================================
# HISTORICAL STORAGE
# ============================================

HISTORICAL_ROOT = "historical"

# ============================================
# MATCHING SETTINGS
# ============================================

MIN_MATCH_SCORE = 0.25
MAX_TIME_DIFF_MINUTES = 60 * 24 * 7  # 7 days

# ============================================
# INDEX SETTINGS
# ============================================

INDEX_VOL_WINDOW = 20
INDEX_MOMENTUM_SHORT = 20
INDEX_MOMENTUM_LONG = 60
INDEX_LEAD_LAG_MAX_DAYS = 5

# ============================================
# TIMESERIES SIGNAL SETTINGS
# ============================================

FUTURE_HORIZON_DAYS = [1, 3, 5]
MIN_SIGNAL_MATCH_SCORE = 0.20

# ============================================
# DEBUG
# ============================================

DEBUG_MODE = True


def print_config():
    print("\n=== CONFIG ===")
    print(f"GLOBAL_START_DATE: {GLOBAL_START_DATE}")
    print(f"FINNHUB KEY SET: {bool(FINNHUB_API_KEY)}")
    print(f"GDELT KEY SET: {bool(GDELT_API_KEY)}")
    print(f"POLY_GAMMA_BASE: {POLY_GAMMA_BASE}")
    print(f"POLY_CLOB_BASE: {POLY_CLOB_BASE}")
    print(f"KALSHI_BASE: {KALSHI_BASE}")
    print(f"DEBUG_MODE: {DEBUG_MODE}")