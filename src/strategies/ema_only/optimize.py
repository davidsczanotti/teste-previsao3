from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import optuna
import pandas as pd

from ...utils.data_loader import load_data
from .backtest import backtest_ema_only, EmaOnlyParams


def _time_splits(df: pd.DataFrame, k: int) -> List[Tuple[int, int]]:
    """Return k contiguous [start, end) index ranges covering df without overlap."""
    n = len(df)
    if k <= 0:
        return [(0, n)]
    base = n // k
    rem = n % k
    splits: List[Tuple[int, int]] = []
    start = 0
    for i in range(k):
        length = base + (1 if i < rem else 0)
        end = start + length
        splits.append((start, end))
        start = end
    return splits


def make_objective(
    df_train: pd.DataFrame,
    lot_size: float,
    fee_rate: float,
    wfa_splits: int,
    penalty_per_trade: float,
):
    def objective(trial: optuna.Trial) -> float:
        ema_period = trial.suggest_int("ema_period", 5, 200)
        use_cross = trial.suggest_categorical("use_cross", [False, True])

        try:
            total = 0.0
            n_trades_total = 0
            for s, e in _time_splits(df_train, max(1, wfa_splits)):
                seg = df_train.iloc[s:e].copy()
                # Guard for too-small segments
                if len(seg) < ema_period + 5:
                    continue
                _, pnl, stats = backtest_ema_only(
                    seg,
                    params=EmaOnlyParams(
                        ema_period=ema_period,
                        lot_size=lot_size,
                        fee_rate=fee_rate,
                        use_cross=use_cross,
                    ),
                    initial_capital=1_000.0,
                )
                total += float(pnl)
                n_trades_total += int(stats.get("num_trades", 0))

            score = total - penalty_per_trade * n_trades_total
            # If no trades at all across splits, discourage solution
            if n_trades_total == 0:
                score -= 1.0
            return score
        except Exception:
            return -1e12

    return objective


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna optimization for EMA-only strategy with time-split CV")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--wfa-splits", type=int, default=5)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lot-size", type=float, default=0.001)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--penalty-per-trade", type=float, default=0.0)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    # Load dataset
    df_all = load_data(args.symbol, args.interval, days=args.days, use_cache_only=args.cache_only)
    n = len(df_all)
    split = int(n * args.train_frac)
    df_train = df_all.iloc[:split].copy()
    df_valid = df_all.iloc[split:].copy()

    print(f"EMA-only optimize: {args.symbol} {args.interval} days={args.days} | N={n} train={len(df_train)} valid={len(df_valid)}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(
            df_train=df_train,
            lot_size=args.lot_size,
            fee_rate=args.fee_rate,
            wfa_splits=args.wfa_splits,
            penalty_per_trade=args.penalty_per_trade,
        ),
        n_trials=args.trials,
    )

    print("Best params (train splits):", study.best_params)
    print("Best objective:", study.best_value)

    bp = study.best_params

    # Evaluate on full train and holdout valid
    _, pnl_tr, stats_tr = backtest_ema_only(
        df_train.copy(),
        params=EmaOnlyParams(
            ema_period=int(bp["ema_period"]),
            lot_size=args.lot_size,
            fee_rate=args.fee_rate,
            use_cross=bool(bp["use_cross"]),
        ),
        initial_capital=1_000.0,
    )
    _, pnl_val, stats_val = backtest_ema_only(
        df_valid.copy(),
        params=EmaOnlyParams(
            ema_period=int(bp["ema_period"]),
            lot_size=args.lot_size,
            fee_rate=args.fee_rate,
            use_cross=bool(bp["use_cross"]),
        ),
        initial_capital=1_000.0,
    )

    rec = {
        "strategy": "ema_only",
        "symbol": args.symbol,
        "interval": args.interval,
        "days": args.days,
        "train_frac": args.train_frac,
        "wfa_splits": args.wfa_splits,
        "lot_size": args.lot_size,
        "fee_rate": args.fee_rate,
        "penalty_per_trade": args.penalty_per_trade,
        "best_params": {
            "ema_period": int(bp["ema_period"]),
            "use_cross": bool(bp["use_cross"]),
        },
        "train_metrics": stats_tr,
        "valid_metrics": stats_val,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = f"ema_only_optuna_{args.symbol}_{args.interval}_{ts}"
    json_path = outdir / f"{base}.json"
    json_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save lightweight active config
    active_dir = outdir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)
    active_path = active_dir / f"ema_only_{args.symbol}_{args.interval}.json"
    active_path.write_text(json.dumps({
        "strategy": "ema_only",
        "symbol": args.symbol,
        "interval": args.interval,
        "best_params": rec["best_params"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Active config updated: {active_path}")


if __name__ == "__main__":
    main()

