import os

START_DATE = os.getenv("START_DATE", "2021-08-01")
END_DATE = os.getenv("END_DATE", "2026-04-09")

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "SPY,QQQ,^GDAXI,^VIX,GLD,USO").split(",") if s.strip()]

DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"

MAX_POLYMARKETS = int(os.getenv("MAX_POLYMARKETS", "200"))
MAX_KALSHI_MARKETS = int(os.getenv("MAX_KALSHI_MARKETS", "200"))

POLY_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLY_CLOB_BASE = "https://clob.polymarket.com"

KALSHI_BASE = os.getenv("KALSHI_BASE", "https://api.elections.kalshi.com/trade-api/v2")

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "0.15"))

MIN_PAIR_SCORE = float(os.getenv("MIN_PAIR_SCORE", "0.45"))
MAX_MATCHED_PAIRS = int(os.getenv("MAX_MATCHED_PAIRS", "300"))