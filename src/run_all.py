from utils import ensure_dirs
from fetch_market_prices import fetch_market_data
from fetch_polymarket import fetch_polymarket_markets, fetch_polymarket_daily_prices
from fetch_kalshi import fetch_kalshi_markets, fetch_kalshi_daily_candles
from build_dataset import build_final_dataset


def run_all() -> None:
    ensure_dirs()

    fetch_market_data()

    poly_markets = fetch_polymarket_markets()
    if not poly_markets.empty:
        fetch_polymarket_daily_prices(poly_markets)

    kalshi_markets = fetch_kalshi_markets()
    if not kalshi_markets.empty:
        fetch_kalshi_daily_candles(kalshi_markets)

    build_final_dataset()
    print("Pipeline complete.")


if __name__ == "__main__":
    run_all()