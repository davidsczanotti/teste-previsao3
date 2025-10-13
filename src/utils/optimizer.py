#!/usr/bin/env python3
"""
Generic Strategy Optimization Module
Implements a reusable optimization framework using Optuna for strategy parameter tuning.
"""

import argparse
import logging
from datetime import datetime, timedelta, UTC
from typing import Callable, Type

import optuna
import pandas as pd
from pydantic import BaseModel

from ..binance_client import get_historical_klines

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(ticker: str, interval: str, days: int) -> pd.DataFrame:
    """Loads historical data from Binance."""
    start_dt = datetime.now(UTC) - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    df = get_historical_klines(ticker, interval, start_str)
    if df.empty:
        raise RuntimeError("No data returned from Binance.")
    return df.sort_values("Date").reset_index(drop=True)


def print_summary(title: str, trades: list, pnl: float):
    """Prints a detailed summary of a backtest."""
    print(f"\n--- {title} ---")

    closed_trades = [t for t in trades if "pnl" in t]
    num_trades = len(closed_trades)

    if num_trades == 0:
        print("No closed trades in the period.")
        print(f"Final P&L: $ {pnl:.2f}")
        return

    wins = [t for t in closed_trades if t["pnl"] > 0]
    total_profit = sum(t["pnl"] for t in wins)
    total_loss = abs(sum(t["pnl"] for t in closed_trades if t["pnl"] < 0))

    win_rate = (len(wins) / num_trades) * 100 if num_trades > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    print(
        f"P&L Final: $ {pnl:.2f} | Trades: {num_trades} | Win Rate: {win_rate:.2f}% | Profit Factor: {profit_factor:.2f}"
    )


def run_optimization_cli(
    strategy_name: str,
    default_symbol: str,
    default_timeframe: str,
    objective_func_creator: Callable,
    backtest_func: Callable,
    plot_func: Callable,
    config_model: Type[BaseModel],
    save_config_func: Callable,
):
    """
    Sets up and runs the CLI for strategy optimization.
    """
    parser = argparse.ArgumentParser(description=f"Optimize the {strategy_name} strategy with Optuna.")
    parser.add_argument("--ticker", type=str, default=default_symbol, help="Asset symbol.")
    parser.add_argument("--interval", type=str, default=default_timeframe, help="Candle timeframe.")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data.")
    parser.add_argument(
        "--train-frac", type=float, default=0.8, help="Fraction of data for training (e.g., 0.8 for 80%)."
    )
    parser.add_argument("--lot-size", type=float, default=0.1, help="Lot size for trades.")
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--min-trades", type=int, default=20, help="Minimum trade threshold for the objective function."
    )
    args = parser.parse_args()

    logger.info(f"Loading data: {args.ticker} @ {args.interval} for {args.days} days...")
    df = load_data(args.ticker, args.interval, args.days)

    n = len(df)
    split_idx = int(n * args.train_frac)
    df_train = df.iloc[:split_idx].copy()
    df_valid = df.iloc[split_idx:].copy()

    logger.info(f"Total candles: {n} | Training: {len(df_train)} | Validation: {len(df_valid)}")

    study_name = f"{strategy_name.lower()}-{args.ticker}-{args.interval}"
    storage_name = "sqlite:///data/optuna_studies.db"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction="maximize", sampler=sampler, load_if_exists=True
    )

    objective = objective_func_creator(df_train, args.lot_size, args.min_trades)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True, gc_after_trial=True)

    print("\n--- Optimization Finished ---")
    print(f"Best score: {study.best_value:.2f}")
    print("Best parameters found:")
    print(study.best_params)

    best_config_data = {
        "ticker": args.ticker,
        "interval": args.interval,
        "days": args.days,
        "lot_size": args.lot_size,
        **study.best_params,
    }
    best_config = config_model(**best_config_data)
    active_path = save_config_func(best_config)
    print(f"\nActive configuration saved to: {active_path}")

    # Backtest on training data
    trades_train, pnl_train, _ = backtest_func(
        df_train.copy(),
        **study.best_params,
        lot_size=best_config.lot_size,
    )
    print_summary("In-Sample / Training Results", trades_train, pnl_train)

    # Backtest on validation data
    trades_valid, pnl_valid, df_valid_indicators = backtest_func(
        df_valid.copy(),
        **study.best_params,
        lot_size=best_config.lot_size,
    )
    print_summary("Out-of-Sample / Validation Results", trades_valid, pnl_valid)

    if not df_valid.empty and plot_func:
        print("\nGenerating chart for the validation period...")
        plot_func(df_valid_indicators, trades_valid, f"al_brooks_backtest_{args.ticker}_validation.png")
