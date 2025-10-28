from __future__ import annotations

"""
Populate the local klines cache (data/klines_cache.db) for a given symbol/timeframe.

Usage (from repo root):

  poetry run python -m scripts.populate_cache BTCUSDT 1h --start "2017-01-01 00:00:00"

Notes
- Requires network access and BINANCE_OFFLINE must NOT be set to 1.
- When --start/--end are provided and parseable, the loader will funnel data
  through the caching layer and persist into data/klines_cache.db.
- If only --days is provided, it computes a start timestamp accordingly.
"""

import argparse
import os
from datetime import datetime, timedelta, UTC

from src.binance_client import get_historical_klines


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Populate local klines cache (SQLite)")
    ap.add_argument("symbol", help="Symbol, e.g., BTCUSDT")
    ap.add_argument("interval", help="Interval, e.g., 1m, 1h, 4h, 1d")
    ap.add_argument("--start", default=None, help="Start datetime (YYYY-MM-DD HH:MM:SS, UTC)")
    ap.add_argument("--end", default=None, help="End datetime (YYYY-MM-DD HH:MM:SS, UTC)")
    ap.add_argument("--days", type=int, default=365, help="If --start not given, go back this many days")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if os.environ.get("BINANCE_OFFLINE", "0") == "1":
        raise SystemExit(
            "BINANCE_OFFLINE=1 detectado. Para popular o cache, remova essa variável (ou defina 0) e rode novamente."
        )

    if args.start is None:
        start_dt = datetime.now(UTC) - timedelta(days=max(1, int(args.days)))
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        start_str = args.start

    print(f"[populate] Baixando {args.symbol} @ {args.interval} desde {start_str} até {args.end or '(agora)'}...")
    df = get_historical_klines(args.symbol, args.interval, start_str, args.end)
    count = 0 if df is None else len(df)
    if count == 0:
        raise SystemExit("Nenhum dado retornado. Verifique símbolo/intervalo/tempo e conectividade.")
    first = df.iloc[0]["Date"].strftime("%Y-%m-%d %H:%M:%S")
    last = df.iloc[-1]["Date"].strftime("%Y-%m-%d %H:%M:%S")
    print(f"[populate] OK: {count} candles persistidos em data/klines_cache.db ({first} -> {last})")


if __name__ == "__main__":
    main()

