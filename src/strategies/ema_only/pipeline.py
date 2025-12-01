from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
import argparse
import numpy as np

from ...utils.data_loader import load_data
from .backtest import EmaOnlyParams, backtest_ema_only


def main() -> None:
    parser = argparse.ArgumentParser(description="EMA-only strategy pipeline (mean reversion by default)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="1m")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--ema-period", type=int, default=8)
    parser.add_argument("--slow-ema-period", type=int, default=None)
    parser.add_argument(
        "--signal-mode",
        type=str,
        default="price_reversion",
        choices=["price_reversion", "ema_cross"],
        help="price_reversion (preço vs EMA) ou ema_cross (cruzamento de EMAs)",
    )
    parser.add_argument("--use-cross", action="store_true", help="Use crossing events instead of simple above/below")
    parser.add_argument("--trend-filter-period", type=int, default=None, help="EMA lenta para filtrar tendência")
    parser.add_argument("--use-trend-filter", action="store_true", help="Ativa filtro de tendência usando trend_filter_period")
    parser.add_argument("--pullback-pct", type=float, default=0.0, help="Exige distância % abaixo da EMA para entrar")
    parser.add_argument("--lot-size", type=float, default=0.001)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--cache-only", action="store_true", help="Use only cached data (no network)")
    args = parser.parse_args()

    # Load data (prefer cache-only if requested)
    df = load_data(args.symbol, args.interval, days=args.days, use_cache_only=args.cache_only)

    params = EmaOnlyParams(
        ema_period=args.ema_period,
        slow_ema_period=args.slow_ema_period,
        trend_filter_period=args.trend_filter_period,
        use_trend_filter=args.use_trend_filter,
        pullback_pct=args.pullback_pct,
        lot_size=args.lot_size,
        fee_rate=args.fee_rate,
        use_cross=args.use_cross,
        signal_mode=args.signal_mode,
    )

    trades, total_pnl, stats = backtest_ema_only(df, params=params, initial_capital=1_000.0)

    print("EMA-only Backtest Summary:")
    print(
        f"Symbol={args.symbol} Interval={args.interval} Days={args.days} Mode={params.signal_mode} EMA={params.ema_period} "
        f"SlowEMA={params.slow_ema_period} TrendFilter={params.use_trend_filter}:{params.trend_filter_period} "
        f"Pullback={params.pullback_pct} UseCross={params.use_cross} Lot={params.lot_size} Fee={params.fee_rate}"
    )
    print(
        "PnL: ${:.2f} | Return: {:.2f}% | Trades: {} | Win rate: {:.2f}% | MDD: {:.2f}%".format(
            stats["pnl"], stats["return_pct"], stats["num_trades"], stats["win_rate"], stats["max_drawdown_pct"]
        )
    )

    # Save JSON summary
    out_dir = os.path.join("reports", "summary")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"ema_only_{args.symbol}_{args.interval}_{ts}.json")

    def _default(o):
        try:
            if isinstance(o, (np.floating, )):
                return float(o)
            if isinstance(o, (np.integer, )):
                return int(o)
            import pandas as pd
            if isinstance(o, (pd.Timestamp, )):
                return o.isoformat()
        except Exception:
            pass
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"params": params.__dict__, "stats": stats, "trades": trades}, f, ensure_ascii=False, indent=2, default=_default)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
