from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
import argparse
import numpy as np

from .config import DeepTripleRsiConfig
from .train import train
from .backtest import run_comprehensive_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Triple RSI Deep - End-to-End Pipeline")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="1m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--use-transformer", action="store_true")
    parser.add_argument("--wfa-splits", type=int, default=5)
    parser.add_argument("--mc-sims", type=int, default=500)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--risk-profile", type=str, default="balanced", choices=["conservative", "balanced", "aggressive"]) 
    parser.add_argument("--trend-intervals", type=str, default="5m,15m,1h")
    # Synthetic data helper
    parser.add_argument("--synthetic-prices", type=str, default=None, help="Comma-separated prices to insert into local cache for the selected symbol/interval before training")
    args = parser.parse_args()

    # Parse trend intervals list (fallback to primary interval in synthetic mode)
    if args.synthetic_prices:
        tfs = [args.interval.strip()]
    else:
        tfs = [s.strip() for s in args.trend_intervals.split(',') if s.strip()]

    cfg = DeepTripleRsiConfig(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        episodes=args.episodes,
        use_transformer=args.use_transformer,
        reward_profile=args.risk_profile,
        trend_intervals=tfs,
    )

    # If synthetic prices provided, insert into cache and prefer deterministic episodes
    if args.synthetic_prices:
        from ...utils.synthetic import parse_prices, store_synthetic_series
        prices = parse_prices(args.synthetic_prices)
        store_synthetic_series(cfg.symbol, cfg.interval, prices)
        # Deterministic training over the whole window for overfit demo
        cfg.random_start = False
        cfg.episode_len = None
        cfg.cache_only = True

    out_dir = os.path.join("reports", "agents", "triple_rsi_deep")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{cfg.symbol}_{cfg.interval}_ppo.pt")

    if args.force_train or not os.path.exists(model_path):
        print("Training model...")
        res = train(cfg, model_path=model_path)
        model_path = res["model_path"]
    else:
        print(f"Using existing model: {model_path}")

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=cfg.days)

    # If synthetic, align backtest period to available cache range to avoid empty data
    if args.synthetic_prices:
        from ...cache.klines_cache import earliest_open_time as _earliest, latest_open_time as _latest
        ems = _earliest(cfg.symbol, cfg.interval)
        lms = _latest(cfg.symbol, cfg.interval)
        if ems is not None and lms is not None:
            start_dt = datetime.utcfromtimestamp(ems / 1000)
            end_dt = datetime.utcfromtimestamp(lms / 1000)
    results = run_comprehensive_backtest(
        model_path=model_path,
        start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        validation_splits=args.wfa_splits,
        num_simulations=args.mc_sims,
        config=cfg,
    )

    summary_dir = os.path.join("reports", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(summary_dir, f"triple_rsi_deep_{cfg.symbol}_{cfg.interval}_{ts}.json")
    def _default(o):
        try:
            if isinstance(o, (np.floating, )):
                return float(o)
            if isinstance(o, (np.integer, )):
                return int(o)
            # pandas Timestamp
            import pandas as pd
            if isinstance(o, (pd.Timestamp, )):
                return o.isoformat()
        except Exception:
            pass
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=_default)
    print(f"Saved pipeline summary to {out_path}")


if __name__ == "__main__":
    main()
