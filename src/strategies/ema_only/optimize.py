from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import optuna
import pandas as pd

from ...utils.data_loader import load_data
from .backtest import backtest_ema_only, EmaOnlyParams, compute_ema


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


def prepare_dataset_with_reference(
    symbol: str,
    timeframe: str,
    days: int,
    use_cache_only: bool,
    ref_timeframe: Optional[str],
    ref_days: Optional[int],
    ref_ema_period: Optional[int],
) -> pd.DataFrame:
    base = load_data(symbol, timeframe, days=days, use_cache_only=use_cache_only).sort_values("Date").reset_index(drop=True)
    if ref_timeframe and ref_ema_period:
        ref_df = load_data(symbol, ref_timeframe, days=ref_days or days, use_cache_only=use_cache_only)
        ref_df = ref_df.sort_values("Date").reset_index(drop=True).copy()
        ref_df["ref_ema"] = compute_ema(ref_df["close"].astype(float), int(ref_ema_period))
        base = pd.merge_asof(base, ref_df[["Date", "ref_ema"]], on="Date", direction="backward")
    return base


def make_objective(
    df_train: pd.DataFrame,
    signal_mode: str,
    slow_ema_period: Optional[int],
    slow_ema_grid: Tuple[int, int] | None,
    ref_filter_enabled: bool,
    ref_timeframe: Optional[str],
    ref_ema_period: Optional[int],
    lot_size: float,
    fee_rate: float,
    wfa_splits: int,
    penalty_per_trade: float,
    objective: str,
    dd_penalty: float,
    ref_buffer_grid: Tuple[float, float] | None = None,
):
    def objective(trial: optuna.Trial) -> float:
        ema_period = trial.suggest_int("ema_period", 5, 200)
        slow_period = trial.suggest_int("slow_ema_period", slow_ema_grid[0], slow_ema_grid[1]) if slow_ema_grid else slow_ema_period
        use_cross = trial.suggest_categorical("use_cross", [False, True])
        ref_buffer = 0.0
        if ref_buffer_grid:
            ref_buffer = trial.suggest_float("ref_buffer_pct", ref_buffer_grid[0], ref_buffer_grid[1])

        try:
            total_pnl = 0.0
            n_trades_total = 0
            max_dds: List[float] = []
            sharpes: List[float] = []
            calmars: List[float] = []
            for s, e in _time_splits(df_train, max(1, wfa_splits)):
                seg = df_train.iloc[s:e].copy()
                # Guard for too-small segments
                if len(seg) < ema_period + 5:
                    continue
                use_cross_param = use_cross if signal_mode != "ema_cross" else False
                _, pnl, stats = backtest_ema_only(
                    seg,
                    params=EmaOnlyParams(
                        ema_period=ema_period,
                        slow_ema_period=slow_period,
                        signal_mode=signal_mode,
                        lot_size=lot_size,
                        fee_rate=fee_rate,
                        use_cross=use_cross_param,
                        ref_filter_enabled=ref_filter_enabled or bool(ref_buffer_grid),
                        ref_buffer_pct=ref_buffer,
                        ref_timeframe=ref_timeframe,
                        ref_ema_period=ref_ema_period,
                    ),
                    initial_capital=1_000.0,
                )
                total_pnl += float(pnl)
                n_trades_total += int(stats.get("num_trades", 0))
                max_dds.append(abs(float(stats.get("max_drawdown_pct", 0.0))))
                sharpes.append(float(stats.get("sharpe", 0.0)))
                calmars.append(float(stats.get("calmar", 0.0)))

            # If no trades at all across splits, discourage solution
            if n_trades_total == 0:
                return -1e12

            dd_pen = dd_penalty * max(max_dds) if max_dds else 0.0

            if objective == "pnl":
                score = total_pnl - penalty_per_trade * n_trades_total - dd_pen
            elif objective == "sharpe":
                score = float(np.nanmean(sharpes)) if sharpes else -1e12
            elif objective == "calmar":
                score = float(np.nanmean(calmars)) if calmars else -1e12
            else:  # combo
                score = (
                    total_pnl
                    - penalty_per_trade * n_trades_total
                    - dd_pen
                    + 50.0 * float(np.nanmean(sharpes)) if sharpes else total_pnl
                )
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
    ap.add_argument("--signal-mode", type=str, default="ema_cross", choices=["ema_cross", "price_reversion"])
    ap.add_argument("--slow-ema-period", type=int, default=None, help="usado em ema_cross")
    ap.add_argument("--slow-ema-min", type=int, default=None, help="se informado junto com --slow-ema-max, otimiza slow_ema_period")
    ap.add_argument("--slow-ema-max", type=int, default=None, help="se informado junto com --slow-ema-min, otimiza slow_ema_period")
    ap.add_argument("--lot-size", type=float, default=0.001)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--penalty-per-trade", type=float, default=0.0)
    ap.add_argument("--objective", type=str, default="pnl", choices=["pnl", "sharpe", "calmar", "combo"])
    ap.add_argument("--dd-penalty", type=float, default=0.0, help="penaliza drawdown absoluto (%) nas metas pnl/combo")
    ap.add_argument("--ref-timeframe", type=str, default=None, help="timeframe de referência para viés (ex.: 1d)")
    ap.add_argument("--ref-days", type=int, default=None, help="dias para carregar no TF de referência")
    ap.add_argument("--ref-ema-period", type=int, default=None, help="período da EMA no TF de referência")
    ap.add_argument("--ref-buffer-min", type=float, default=None, help="limite inferior para ref_buffer_pct (ativa otimização de buffer)")
    ap.add_argument("--ref-buffer-max", type=float, default=None, help="limite superior para ref_buffer_pct (ativa otimização de buffer)")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    # Load dataset with optional reference EMA
    df_all = prepare_dataset_with_reference(
        symbol=args.symbol,
        timeframe=args.interval,
        days=args.days,
        use_cache_only=args.cache_only,
        ref_timeframe=args.ref_timeframe,
        ref_days=args.ref_days,
        ref_ema_period=args.ref_ema_period,
    )
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
            signal_mode=args.signal_mode,
            slow_ema_period=args.slow_ema_period,
            slow_ema_grid=(args.slow_ema_min, args.slow_ema_max) if args.slow_ema_min is not None and args.slow_ema_max is not None else None,
            ref_filter_enabled=bool(args.ref_timeframe and args.ref_ema_period),
            ref_timeframe=args.ref_timeframe,
            ref_ema_period=args.ref_ema_period,
            lot_size=args.lot_size,
            fee_rate=args.fee_rate,
            wfa_splits=args.wfa_splits,
            penalty_per_trade=args.penalty_per_trade,
            objective=args.objective,
            dd_penalty=args.dd_penalty,
            ref_buffer_grid=(args.ref_buffer_min, args.ref_buffer_max) if args.ref_buffer_min is not None and args.ref_buffer_max is not None else None,
        ),
        n_trials=args.trials,
    )

    print("Best params (train splits):", study.best_params)
    print("Best objective:", study.best_value)

    bp = study.best_params

    # Evaluate on full train and holdout valid
    slow_eval = int(bp["slow_ema_period"]) if "slow_ema_period" in bp else args.slow_ema_period

    _, pnl_tr, stats_tr = backtest_ema_only(
        df_train.copy(),
        params=EmaOnlyParams(
            ema_period=int(bp["ema_period"]),
            slow_ema_period=slow_eval,
            signal_mode=args.signal_mode,
            lot_size=args.lot_size,
            fee_rate=args.fee_rate,
            use_cross=bool(bp["use_cross"]) if args.signal_mode != "ema_cross" else False,
            ref_filter_enabled=bool(args.ref_timeframe and args.ref_ema_period),
            ref_buffer_pct=float(bp["ref_buffer_pct"]) if "ref_buffer_pct" in bp else 0.0,
            ref_timeframe=args.ref_timeframe,
            ref_ema_period=args.ref_ema_period,
        ),
        initial_capital=1_000.0,
    )
    _, pnl_val, stats_val = backtest_ema_only(
        df_valid.copy(),
        params=EmaOnlyParams(
            ema_period=int(bp["ema_period"]),
            slow_ema_period=slow_eval,
            signal_mode=args.signal_mode,
            lot_size=args.lot_size,
            fee_rate=args.fee_rate,
            use_cross=bool(bp["use_cross"]) if args.signal_mode != "ema_cross" else False,
            ref_filter_enabled=bool(args.ref_timeframe and args.ref_ema_period),
            ref_buffer_pct=float(bp["ref_buffer_pct"]) if "ref_buffer_pct" in bp else 0.0,
            ref_timeframe=args.ref_timeframe,
            ref_ema_period=args.ref_ema_period,
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
        "objective": args.objective,
        "dd_penalty": args.dd_penalty,
        "signal_mode": args.signal_mode,
        "slow_ema_period": args.slow_ema_period,
        "slow_ema_min": args.slow_ema_min,
        "slow_ema_max": args.slow_ema_max,
        "ref_timeframe": args.ref_timeframe,
        "ref_days": args.ref_days,
        "ref_ema_period": args.ref_ema_period,
        "ref_buffer_min": args.ref_buffer_min,
        "ref_buffer_max": args.ref_buffer_max,
        "best_params": {
            "ema_period": int(bp["ema_period"]),
            "use_cross": bool(bp["use_cross"]),
            "ref_buffer_pct": float(bp.get("ref_buffer_pct")) if "ref_buffer_pct" in bp else None,
            "slow_ema_period": int(bp.get("slow_ema_period")) if "slow_ema_period" in bp else args.slow_ema_period,
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
