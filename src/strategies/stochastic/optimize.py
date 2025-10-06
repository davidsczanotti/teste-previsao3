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

# Evita abrir janelas de gráfico durante a otimização
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("STOCH_BACKTEST_PLOT", "0")
matplotlib.use("Agg", force=True)
plt.show = lambda *args, **kwargs: None

from ...binance_client import get_historical_klines
from .backtest import backtest_stochastic, StochParams
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
        # Espaços de busca e restrições
        k = trial.suggest_int("k", 5, 30)
        oversold = trial.suggest_int("oversold", 10, 30)
        overbought = trial.suggest_int("overbought", max(oversold + 40, 70), 90)
        d_period = trial.suggest_int("d_period", 2, 5)
        use_kd_cross = trial.suggest_categorical("use_kd_cross", [True, False])
        enable_trend = trial.suggest_categorical("enable_trend", [True, False])
        ema_period = trial.suggest_int("ema_period", 20, 60) if enable_trend else 0
        enable_adx = trial.suggest_categorical("enable_adx", [True, False])
        adx_period = trial.suggest_int("adx_period", 10, 20) if enable_adx else 14
        min_adx = trial.suggest_int("min_adx", 15, 35) if enable_adx else 20
        confirm_bars = trial.suggest_int("confirm_bars", 0, 1)
        cooldown_bars = trial.suggest_int("cooldown_bars", 0, 2)
        min_hold_bars = trial.suggest_int("min_hold_bars", 0, 2)
        penalty_per_trade = trial.suggest_float("penalty_per_trade", 0.0, 0.1)

        try:
            trades, pnl, stats = backtest_stochastic(
                df_train.copy(),
                params=StochParams(
                    k_period=k,
                    oversold=oversold,
                    overbought=overbought,
                    d_period=d_period,
                    use_kd_cross=use_kd_cross,
                    ema_period=(ema_period if enable_trend else None),
                    confirm_bars=confirm_bars,
                    cooldown_bars=cooldown_bars,
                    min_hold_bars=min_hold_bars,
                    use_adx=enable_adx,
                    adx_period=adx_period,
                    min_adx=min_adx,
                ),
                initial_capital=10_000.0,
                lot_size=lot_size,
                fee_rate=fee_rate,
            )
        except Exception:
            return -1e12

        closed = stats.get("num_trades", 0)
        penalty = penalty_per_trade * closed
        return pnl - penalty

    return objective


def summarize(trades, pnl, initial_cap=10_000.0):
    closed = [t for t in trades if "pnl" in t]
    n = len(closed)
    wins = len([t for t in closed if t["pnl"] > 0])
    win_rate = (wins / n * 100.0) if n else 0.0
    return {
        "pnl": float(pnl),
        "num_trades": n,
        "win_rate": win_rate,
        "return_pct": (float(pnl) / initial_cap) * 100.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize Stochastic 25/75 with Optuna (Binance)")
    parser.add_argument("--ticker", default="BTCUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--lot_size", type=float, default=0.001)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="reports")
    parser.add_argument("--fee_rate", type=float, default=0.001, help="Taxa por lado (ex.: 0.001 = 0.1%)")
    args = parser.parse_args()

    print(f"Carregando dados: {args.ticker} @ {args.interval} por {args.days} dias...")
    df = load_data(args.ticker, args.interval, args.days)
    n = len(df)
    split = int(n * args.train_frac)
    df_train = df.iloc[:split].copy()
    df_valid = df.iloc[split:].copy()
    print(f"Total candles: {n} | Treino: {len(df_train)} | Validação: {len(df_valid)}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(df_train, args.lot_size, args.fee_rate), n_trials=args.trials)

    print("\nMelhores parâmetros (treino):")
    print(study.best_params)
    print(f"Melhor P&L (treino): $ {study.best_value:.2f}")

    p = study.best_params

    trades_tr, pnl_tr, stats_tr = backtest_stochastic(
        df_train.copy(),
        params=StochParams(
            k_period=p["k"],
            oversold=p["oversold"],
            overbought=p["overbought"],
            d_period=p.get("d_period", 3),
            use_kd_cross=p.get("use_kd_cross", True),
            ema_period=(p.get("ema_period") or None) if p.get("enable_trend", False) else None,
            confirm_bars=p.get("confirm_bars", 0),
            cooldown_bars=p.get("cooldown_bars", 0),
            min_hold_bars=p.get("min_hold_bars", 0),
            use_adx=p.get("enable_adx", False),
            adx_period=p.get("adx_period", 14),
            min_adx=p.get("min_adx", 20),
        ),
        lot_size=args.lot_size,
        fee_rate=args.fee_rate,
    )
    trades_val, pnl_val, stats_val = backtest_stochastic(
        df_valid.copy(),
        params=StochParams(
            k_period=p["k"],
            oversold=p["oversold"],
            overbought=p["overbought"],
            d_period=p.get("d_period", 3),
            use_kd_cross=p.get("use_kd_cross", True),
            ema_period=(p.get("ema_period") or None) if p.get("enable_trend", False) else None,
            confirm_bars=p.get("confirm_bars", 0),
            cooldown_bars=p.get("cooldown_bars", 0),
            min_hold_bars=p.get("min_hold_bars", 0),
            use_adx=p.get("enable_adx", False),
            adx_period=p.get("adx_period", 14),
            min_adx=p.get("min_adx", 20),
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
            "k_period": p["k"],
            "oversold": p["oversold"],
            "overbought": p["overbought"],
            "d_period": p.get("d_period", 3),
            "use_kd_cross": p.get("use_kd_cross", True),
            "ema_period": (p.get("ema_period") if p.get("enable_trend", False) else None),
            "confirm_bars": p.get("confirm_bars", 0),
            "cooldown_bars": p.get("cooldown_bars", 0),
            "min_hold_bars": p.get("min_hold_bars", 0),
            "enable_adx": p.get("enable_adx", False),
            "adx_period": p.get("adx_period", 14),
            "min_adx": p.get("min_adx", 20),
            "lot_size": args.lot_size,
        },
        "train_metrics": stats_tr,
        "valid_metrics": stats_val,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = f"stoch_optuna_{args.ticker}_{args.interval}_{ts}"

    json_path = outdir / f"{base}.json"
    json_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = outdir / f"{base}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Stochastic Optimization Report\n\n")
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
        f.write("\n## How to Apply\n")
        f.write("Live example (src/strategies/stochastic/live.py):\n\n")
        f.write(
            """
live_stochastic(
    ticker="{ticker}",
    interval_minutes={interval_minutes},
    initial_capital=1000.0,
    lot_size={lot},
    params=StochParams(k_period={k}, oversold={os}, overbought={ob}),
)
""".format(
                ticker=args.ticker,
                interval_minutes=5 if args.interval.endswith("m") else 15,
                lot=args.lot_size,
                k=p["k"],
                os=p["oversold"],
                ob=p["overbought"],
            )
        )
        f.write("\nBacktest example (src/strategies/stochastic/backtest.py):\n\n")
        f.write(
            """
from datetime import datetime, timedelta, UTC
from src.binance_client import get_historical_klines
from src.strategies.stochastic.backtest import backtest_stochastic, StochParams

start_dt = datetime.now(UTC) - timedelta(days={days})
start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
df = get_historical_klines("{ticker}", "{interval}", start_str)
trades, pnl, stats = backtest_stochastic(
    df, params=StochParams(k_period={k}, oversold={os}, overbought={ob}), lot_size={lot}
)
print(pnl, stats)
""".format(
                days=args.days,
                ticker=args.ticker,
                interval=args.interval,
                k=p["k"],
                os=p["oversold"],
                ob=p["overbought"],
                lot=args.lot_size,
            )
        )

    # Atualiza configuração ativa p/ consumo em live/backtest
    active_path = save_active_config(rec, reports_dir=args.outdir)
    print(f"\nRelatórios salvos em: {json_path} e {md_path}")
    print(f"Config ativa (Stoch) atualizada: {active_path}")


if __name__ == "__main__":
    main()
