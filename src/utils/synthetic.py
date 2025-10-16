from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import List

from ..cache.klines_cache import store_rows


def parse_prices(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [float(p) for p in parts]


def store_synthetic_series(symbol: str, interval: str, prices: List[float], align_to_now: bool = True) -> None:
    if not prices:
        raise ValueError("Empty prices list")

    # Determine start time so that the latest candle ends exactly 1 interval before 'now'
    now_ms = int(time.time() * 1000)
    if interval.endswith('m'):
        mult = int(interval[:-1]) * 60_000
    elif interval.endswith('h'):
        mult = int(interval[:-1]) * 3_600_000
    elif interval.endswith('d'):
        mult = int(interval[:-1]) * 86_400_000
    else:
        raise ValueError(f"Unsupported interval: {interval}")

    last_open_ms = now_ms - mult  # last candle open one interval before now
    start_ms = last_open_ms - (len(prices) - 1) * mult

    rows = []
    for i, p in enumerate(prices):
        t = start_ms + i * mult
        o = float(p)
        h = o + 0.1
        l = o - 0.1
        c = o
        v = 1.0
        rows.append((t, o, h, l, c, v))
    store_rows(symbol, interval, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Insert synthetic OHLCV series into local cache")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--prices", required=True, help="Comma-separated prices, e.g. 30,31,32,33,34")
    args = ap.parse_args()

    prices = parse_prices(args.prices)
    store_synthetic_series(args.symbol.strip().upper(), args.interval.strip(), prices)
    print(f"Inserted {len(prices)} candles for {args.symbol}@{args.interval}")


if __name__ == "__main__":
    main()

