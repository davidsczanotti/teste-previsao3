#!/usr/bin/env python3
"""
Utility CLI to update the local klines cache (data/klines_cache.db).

Example:
  poetry run python -m src.utils.update_cache --symbol BTCUSDT --interval 1m --days 365

This will fetch missing candles from Binance (respecting your proxy env)
and persist them to the local SQLite cache. No output files are created; the
cache-backed readers will pick up the data afterwards.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, UTC

from ..binance_client import get_historical_klines


def main() -> None:
    ap = argparse.ArgumentParser(description="Update local klines cache via Binance API")
    ap.add_argument("--symbol", required=True, help="Symbol, e.g., BTCUSDT")
    ap.add_argument("--interval", required=True, help="Interval, e.g., 1m, 5m, 1h")
    ap.add_argument("--days", type=int, default=365, help="Days of history to ensure in cache")
    args = ap.parse_args()

    start_dt = datetime.now(UTC) - timedelta(days=args.days)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    df = get_historical_klines(args.symbol, args.interval, start_str)
    if df.empty:
        print("No data fetched. Check symbol/interval/network/proxy.")
        return
    print(f"Cache updated: {args.symbol}@{args.interval} | candles: {len(df)} | range: {df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}")


if __name__ == "__main__":
    main()

