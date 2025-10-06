from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import optuna

from .backtest import backtest_ma_crossover, BacktestResult
from .config import MaCrossoverBacktestConfig
from ...binance_client import get_historical_klines


def run_backtest(cfg: MaCrossoverBacktestConfig) -> BacktestResult:
    from datetime import UTC, timedelta

    start_dt = datetime.now(UTC) - timedelta(days=int(cfg.days))
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    df = get_historical_klines(cfg.ticker, cfg.interval, start_str)
    if df.empty:
        raise RuntimeError("Sem dados para otimização")
    return backtest_ma_crossover(
        df,
        lot_size=cfg.lot_size,
        fee_rate=cfg.fee_rate,
        min_hold_bars=cfg.min_hold_bars,
        cooldown_bars=cfg.cooldown_bars,
        allow_short=cfg.allow_short,
        initial_capital=cfg.lot_size * 100_000,
        ma_short_window=cfg.ma_short_window,
        ma_mid_window=cfg.ma_mid_window,
        ma_long_window=cfg.ma_long_window,
        ma_type=cfg.ma_type,
    )


def objective(trial: optuna.Trial, base_cfg: MaCrossoverBacktestConfig) -> float:
    # sample MA windows ensuring order
    ma_short = trial.suggest_int("ma_short", 5, 20)
    ma_mid = trial.suggest_int("ma_mid", ma_short + 5, 90)
    ma_long = trial.suggest_int("ma_long", ma_mid + 10, 240)

    min_hold = trial.suggest_int("min_hold_bars", 0, 8)
    cooldown = trial.suggest_int("cooldown_bars", 0, 6)

    cfg = MaCrossoverBacktestConfig(
        ticker=base_cfg.ticker,
        interval=base_cfg.interval,
        days=base_cfg.days,
        lot_size=base_cfg.lot_size,
        fee_rate=base_cfg.fee_rate,
        min_hold_bars=min_hold,
        cooldown_bars=cooldown,
        allow_short=base_cfg.allow_short,
        ma_short_window=ma_short,
        ma_mid_window=ma_mid,
        ma_long_window=ma_long,
        ma_type=base_cfg.ma_type,
    )

    res = run_backtest(cfg)
    pnl = res.stats["pnl"]
    trades = max(res.stats["num_trades"], 1)
    dd = abs(res.stats["max_drawdown_pct"])

    score = pnl - 2.0 * trades - 10.0 * dd
    trial.set_user_attr("stats", res.stats)
    trial.set_user_attr("params_cfg", cfg.__dict__)
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna optimizer for MA crossover params")
    parser.add_argument("--ticker", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--lot_size", type=float, default=0.001)
    parser.add_argument("--fee_rate", type=float, default=0.001)
    parser.add_argument("--allow_short", action="store_true")
    parser.add_argument("--ma_type", choices=["sma", "ema"], default="sma")
    args = parser.parse_args()

    base_cfg = MaCrossoverBacktestConfig(
        ticker=args.ticker,
        interval=args.interval,
        days=args.days,
        lot_size=args.lot_size,
        fee_rate=args.fee_rate,
        allow_short=args.allow_short,
        ma_type=args.ma_type,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, base_cfg), n_trials=args.trials)

    best = study.best_trial
    best_cfg = best.user_attrs["params_cfg"]
    best_stats = best.user_attrs["stats"]

    print("Melhor score:", best.value)
    print("Melhor configuração:")
    print(json.dumps(best_cfg, indent=2))
    print("Estatísticas:")
    print(json.dumps(best_stats, indent=2))

    out_dir = Path("reports/ma_crossover_optuna")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"study_{args.ticker}_{args.interval}_{ts}.json").write_text(
        json.dumps(
            {
                "best_score": best.value,
                "best_cfg": best_cfg,
                "best_stats": best_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
