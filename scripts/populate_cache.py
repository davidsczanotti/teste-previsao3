from __future__ import annotations

import argparse

import pandas as pd

from src.binance_client import get_historical_klines
from src.cache.klines_cache import latest_open_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate local Kline cache up to latest available data.")
    parser.add_argument("symbol", nargs="?", default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)")
    parser.add_argument("interval", nargs="?", default="15m", help="Interval (default: 15m)")
    parser.add_argument(
        "--start",
        dest="start",
        default="2017-01-01 00:00:00",
        help="Start datetime (UTC). Binance spot history begins around 2017-07-01.",
    )
    args = parser.parse_args()

    # Verifica o último registro no cache ANTES de qualquer coisa
    latest_before_ms = latest_open_time(args.symbol, args.interval)

    # Trigger fetch+cache by calling the existing client function
    print(f"Fetching {args.symbol} {args.interval} from {args.start} to now...")
    df = get_historical_klines(args.symbol, args.interval, args.start)

    # Verifica o último registro no cache DEPOIS da chamada
    latest_after_ms = latest_open_time(args.symbol, args.interval)

    if df.empty:
        print("No data retrieved. Verify symbol/interval/start date.")
    elif latest_before_ms == latest_after_ms:
        latest_dt = pd.to_datetime(latest_after_ms, unit="ms", utc=True)
        print(f"Cache is already up-to-date. Latest data is from {latest_dt}.")
    else:
        start_dt = pd.to_datetime(latest_before_ms, unit="ms", utc=True) if latest_before_ms else args.start
        end_dt = pd.to_datetime(latest_after_ms, unit="ms", utc=True)
        print(f"Cache updated. Now contains data from {df['Date'].iloc[0]} to {end_dt}.")


if __name__ == "__main__":
    main()
