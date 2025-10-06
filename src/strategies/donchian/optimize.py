from __future__ import annotations

import argparse
from datetime import datetime, timedelta, UTC
from pathlib import Path
import json

import optuna
import os
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)
plt.show = lambda *args, **kwargs: None

from ...binance_client import get_historical_klines
from .backtest import backtest_donchian, DonchianParams
from .config import save_active_config


def load_data(ticker: str, interval: str, days: int) -> pd.DataFrame:
    start_dt = datetime.now(UTC) - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    df = get_historical_klines(ticker, interval, start_str)
    if df.empty:
        raise RuntimeError("Nenhum dado retornado da Binance.")
    return df.sort_values("Date").reset_index(drop=True)


def make_objective(df_train: pd.DataFrame, lot_size: float, fee_rate: float):
    def objective(trial: optuna.Trial) -> float:
        win_h = trial.suggest_int("window_high", 10, 60)
        win_l = trial.suggest_int("window_low", 10, 60)
        atr_p = trial.suggest_int("atr_period", 7, 21)
        atr_x = trial.suggest_float("atr_mult", 1.0, 4.0)
        use_ema = trial.suggest_categorical("use_ema", [True, False])
        ema_p = trial.suggest_int("ema_period", 50, 250) if use_ema else 200
        allow_short = trial.suggest_categorical("allow_short", [False, True])
        penalty_per_trade = trial.suggest_float("penalty_per_trade", 0.0, 0.2)

        try:
            trades, pnl, stats = backtest_donchian(
                df_train.copy(),
                params=DonchianParams(
                    window_high=win_h,
                    window_low=win_l,
                    atr_period=atr_p,
                    atr_mult=atr_x,
                    use_ema=use_ema,
                    ema_period=ema_p,
                    allow_short=allow_short,
                ),
                initial_capital=10_000.0,
                lot_size=lot_size,
                fee_rate=fee_rate,
            )
        except Exception:
            return -1e12

        penalty = penalty_per_trade * stats.get("num_trades", 0)
        return pnl - penalty

    return objective


def main():
    ap = argparse.ArgumentParser(description="Optimize Donchian Breakout (Binance)")
    ap.add_argument("--ticker", default="BTCUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--lot_size", type=float, default=0.001)
    ap.add_argument("--fee_rate", type=float, default=0.0005)
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    print(f"Carregando dados: {args.ticker} @ {args.interval} por {args.days} dias...")
    df = load_data(args.ticker, args.interval, args.days)
    n = len(df)
    split = int(n * args.train_frac)
    df_train = df.iloc[:split].copy()
    df_valid = df.iloc[split:].copy()
    print(f"Total candles: {n} | Treino: {len(df_train)} | Validação: {len(df_valid)}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(make_objective(df_train, args.lot_size, args.fee_rate), n_trials=args.trials)

    print("\nMelhores parâmetros (treino):")
    print(study.best_params)
    print(f"Melhor P&L (treino): $ {study.best_value:.2f}")

    p = study.best_params

    tr_trades, tr_pnl, tr_stats = backtest_donchian(
        df_train.copy(),
        params=DonchianParams(
            window_high=p["window_high"],
            window_low=p["window_low"],
            atr_period=p["atr_period"],
            atr_mult=p["atr_mult"],
            use_ema=p["use_ema"],
            ema_period=(p["ema_period"] if p.get("use_ema", True) else 200),
            allow_short=p["allow_short"],
        ),
        lot_size=args.lot_size,
        fee_rate=args.fee_rate,
    )
    va_trades, va_pnl, va_stats = backtest_donchian(
        df_valid.copy(),
        params=DonchianParams(
            window_high=p["window_high"],
            window_low=p["window_low"],
            atr_period=p["atr_period"],
            atr_mult=p["atr_mult"],
            use_ema=p["use_ema"],
            ema_period=(p["ema_period"] if p.get("use_ema", True) else 200),
            allow_short=p["allow_short"],
        ),
        lot_size=args.lot_size,
        fee_rate=args.fee_rate,
    )

    rec = {
        "ticker": args.ticker,
        "interval": args.interval,
        "days": args.days,
        "train_frac": args.train_frac,
        "lot_size": args.lot_size,
        "fee_rate": args.fee_rate,
        "best_params": {
            "window_high": p["window_high"],
            "window_low": p["window_low"],
            "atr_period": p["atr_period"],
            "atr_mult": p["atr_mult"],
            "use_ema": p["use_ema"],
            "ema_period": (p["ema_period"] if p.get("use_ema", True) else 200),
            "allow_short": p["allow_short"],
            "lot_size": args.lot_size,
        },
        "train_metrics": tr_stats,
        "valid_metrics": va_stats,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = f"donchian_optuna_{args.ticker}_{args.interval}_{ts}"

    (outdir / f"{base}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    with (outdir / f"{base}.md").open("w", encoding="utf-8") as f:
        f.write("# Donchian Optimization Report\n\n")
        f.write(f"- Ticker: {args.ticker}\n")
        f.write(f"- Interval: {args.interval}\n")
        f.write(f"- Days: {args.days}\n")
        f.write(f"- Trials: {args.trials}\n")
        f.write(f"- Train frac: {args.train_frac}\n")
        f.write("\n## Best Parameters\n")
        for k, v in rec["best_params"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Train Metrics\n")
        for k, v in rec["train_metrics"].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Validation Metrics\n")
        for k, v in rec["valid_metrics"].items():
            f.write(f"- {k}: {v}\n")

    active_path = save_active_config(rec, reports_dir=args.outdir)
    print(f"\nRelatórios salvos em: {outdir / (base + '.json')} e {outdir / (base + '.md')}")
    print(f"Config ativa (Donchian) atualizada: {active_path}")


if __name__ == "__main__":
    main()

