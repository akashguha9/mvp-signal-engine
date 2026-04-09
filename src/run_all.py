from utils import ensure_dirs, print_stage
from fetch_market_prices import fetch_market_data
from fetch_polymarket import fetch_polymarket_markets, fetch_polymarket_daily_prices
from fetch_kalshi import fetch_kalshi_markets, fetch_kalshi_daily_candles
from build_dataset import build_final_dataset


def run_all() -> None:
    ensure_dirs()

    print_stage("RUN START")

    fetch_market_data()

    poly_markets = fetch_polymarket_markets()
    if not poly_markets.empty:
        fetch_polymarket_daily_prices(poly_markets)
    else:
        print("Skipping Polymarket daily price fetch because metadata is empty.")

    kalshi_markets = fetch_kalshi_markets()
    if not kalshi_markets.empty:
        fetch_kalshi_daily_candles(kalshi_markets)
    else:
        print("Skipping Kalshi candle fetch because metadata is empty.")

    build_final_dataset()

    print_stage("RUN COMPLETE")


if __name__ == "__main__":
    run_all()