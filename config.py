# config.py

from __future__ import annotations

import os


# ── API KEYS ───────────────────────────────────────────────────────────────────

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d7b971pr01qlbg01e9t0d7b971pr01qlbg01e9tg")
GDELT_API_KEY = os.getenv("GDELT_API_KEY", "")


# ── OUTPUT FILES ───────────────────────────────────────────────────────────────

OUTPUT_EVENT_PANEL = "output_event_panel.csv"
OUTPUT_NEWS_PANEL = "output_news_panel.csv"
OUTPUT_EVENT_MATCHES = "output_event_matches.csv"

OUTPUT_INDEX_HISTORY = "output_index_history.csv"
OUTPUT_INDEX_FEATURES = "output_index_features.csv"
OUTPUT_INDEX_FEATURES_FULL = "output_index_features_full.csv"
OUTPUT_INDEX_MAPPING = "output_index_mapping.csv"
OUTPUT_INDEX_LEADLAG = "output_index_leadlag.csv"
OUTPUT_INDEX_CORRELATIONS = "output_index_correlations.csv"

OUTPUT_PRICE_PANEL = "output_price_panel.csv"
OUTPUT_YAHOO_ENRICH = "output_yahoo_enrich.csv"

OUTPUT_SEED_SUMMARY = "output_seed_summary.csv"

# Historical seed-index panel built by joining seed/index mappings to full
# historical feature rows across timestamps.
OUTPUT_SEED_INDEX_PANEL = "output_seed_index_panel.csv"

# Time-series-ready historical seed-index panel after normalization / cleanup.
OUTPUT_SEED_INDEX_PANEL_TS = "output_seed_index_panel_ts.csv"

OUTPUT_TIMESERIES_DATASET = "output_timeseries_dataset.csv"
OUTPUT_SIGNALS = "output_signals.csv"
OUTPUT_BACKTEST_SUMMARY = "output_backtest_summary.csv"


# ── MODEL / PIPELINE SETTINGS ──────────────────────────────────────────────────

MIN_SIGNAL_MATCH_SCORE = 0.30
FUTURE_HORIZON_DAYS = [1, 3, 5]


# ── MISC ───────────────────────────────────────────────────────────────────────

DEFAULT_TIMEZONE = "UTC"