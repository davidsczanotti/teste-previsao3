from __future__ import annotations

"""
Optimization runner without CLI flags for al_brooks strategy.
Reads settings from src/strategies/al_brooks/config.json (key: "optimize").
Uses cache-only data when configured.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict

import optuna
from optuna.samplers import TPESampler

from .backtest import backtest_al_brooks_inside_bar, plot_backtest
from .config import AlBrooksConfig, save_active_config
from ...utils.data_loader import load_data
from ...utils.optimizer import print_summary
from ...utils.metrics import calculate_metrics


def make_objective(df_train, lot_size: float, min_trade_threshold: int = 20):
    """Creates the objective function for Optuna (in-module version)."""
    threshold = max(1, min_trade_threshold)
    # Constant execution costs during optimization
    FEE_PCT = 0.0004
    SLIPPAGE_PCT = 0.0005

    def objective(trial: optuna.Trial) -> float:
        # Parameter search space
        ema_fast = trial.suggest_int("ema_fast_period", 5, 20)
        ema_medium = trial.suggest_int("ema_medium_period", ema_fast + 3, ema_fast + 25)
        ema_slow = trial.suggest_int("ema_slow_period", ema_medium + 5, ema_medium + 80)

        risk_reward_ratio = trial.suggest_float("risk_reward_ratio", 1.2, 2.0, step=0.1)
        max_avg_deviation_pct = trial.suggest_float("max_avg_deviation_pct", 0.2, 1.0, step=0.05)
        adx_threshold = trial.suggest_float("adx_threshold", 18.0, 28.0, step=1.0)
        atr_stop_multiplier = trial.suggest_float("atr_stop_multiplier", 1.0, 3.0, step=0.1)
        atr_trail_multiplier = trial.suggest_float("atr_trail_multiplier", 0.0, 1.0, step=0.1)
        htf_lookback = trial.suggest_int("htf_lookback", 10, 40)
        min_atr = trial.suggest_float("min_atr", 5.0, 25.0, step=0.5)
        # Inside bar-related toggles and pullback lookback
        use_inside_bar = trial.suggest_categorical("use_inside_bar", [True, False])
        inside_bar_inclusive = trial.suggest_categorical("inside_bar_inclusive", [False, True])
        pullback_lookback = trial.suggest_int("pullback_lookback", 6, 15)

        # Run backtest on training slice
        try:
            trades, pnl, _ = backtest_al_brooks_inside_bar(
                df_train.copy(),
                ema_fast_period=ema_fast,
                ema_medium_period=ema_medium,
                ema_slow_period=ema_slow,
                risk_reward_ratio=risk_reward_ratio,
                max_avg_deviation_pct=max_avg_deviation_pct,
                lot_size=lot_size,
                adx_threshold=adx_threshold,
                atr_stop_multiplier=atr_stop_multiplier,
                atr_trail_multiplier=atr_trail_multiplier,
                htf_lookback=htf_lookback,
                use_inside_bar=use_inside_bar,
                inside_bar_inclusive=inside_bar_inclusive,
                min_atr=min_atr,
                pullback_lookback=pullback_lookback,
                taker_fee_pct=FEE_PCT,
                slippage_pct=SLIPPAGE_PCT,
            )
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return -1e9

        metrics = calculate_metrics(trades)
        trade_count = metrics["total_trades"]
        total_pnl = metrics["total_pnl"]
        profit_factor = metrics["profit_factor"]

        # Hard constraints to avoid low-significance or negative solutions
        if trade_count < threshold:
            return -1e9
        if total_pnl <= 0:
            return -1e9

        if not np.isfinite(profit_factor):
            profit_factor = 10.0

        # Score balances PF and PnL; both already gated to positive
        score = (profit_factor) + (total_pnl / 200.0)
        return score

    return objective


def _load_local_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).resolve().parent / "config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return data


def main() -> None:
    cfg = _load_local_config()
    opt = dict(cfg.get("optimize", {}))

    ticker = opt.get("ticker", "BTCUSDT")
    interval = opt.get("interval", "1m")
    days = int(opt.get("days", 365))
    train_frac = float(opt.get("train_frac", 0.8))
    lot_size = float(opt.get("lot_size", 0.1))
    n_trials = int(opt.get("trials", 50))
    min_trades = int(opt.get("min_trades", 20))
    seed = int(opt.get("seed", 42))
    cache_only = bool(opt.get("cache_only", True))

    print(
        f"Loading data: {ticker} @ {interval} for {days} days... (cache_only={cache_only})"
    )
    df = load_data(ticker, interval, days=days, use_cache_only=cache_only)
    n = len(df)
    split_idx = int(n * train_frac)
    df_train = df.iloc[:split_idx].copy()
    df_valid = df.iloc[split_idx:].copy()
    print(f"Total candles: {n} | Training: {len(df_train)} | Validation: {len(df_valid)}")

    # Build a study name that encodes important search toggles to avoid
    # reusing incompatible past studies and to make results comparable.
    suffix_parts = []
    # These toggles are part of the current search space
    suffix_parts.append("ib")    # inside bar toggles included
    suffix_parts.append("plb")   # pullback_lookback included
    suffix = ("-" + "-".join(suffix_parts)) if suffix_parts else ""

    study_name = f"albrooks-{ticker}-{interval}{suffix}"
    storage = "sqlite:///data/optuna_studies.db"
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    objective = make_objective(df_train, lot_size, min_trade_threshold=min_trades)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)

    print("\n--- Optimization Finished ---")
    print(f"Best score: {study.best_value:.4f}")
    print("Best parameters:")
    print(study.best_params)

    # Evaluate in-sample
    trades_tr, pnl_tr, _ = backtest_al_brooks_inside_bar(
        df_train.copy(), lot_size=lot_size, **study.best_params
    )
    print_summary("In-Sample / Training Results", trades_tr, pnl_tr)

    # Evaluate out-of-sample
    trades_val, pnl_val, df_val_ind = backtest_al_brooks_inside_bar(
        df_valid.copy(), lot_size=lot_size, **study.best_params
    )
    print_summary("Out-of-Sample / Validation Results", trades_val, pnl_val)

    # Gate saving the configuration by validation quality
    met_val = calculate_metrics(trades_val)
    val_trades = met_val.get("total_trades", 0)
    val_pnl = met_val.get("total_pnl", 0.0)
    val_pf = met_val.get("profit_factor", 0.0)
    if val_trades >= min_trades and val_pnl > 0 and (np.isfinite(val_pf) and val_pf >= 1.05):
        best_config_data = {
            "ticker": ticker,
            "interval": interval,
            "days": days,
            "lot_size": lot_size,
            **study.best_params,
        }
        best_config = AlBrooksConfig(**best_config_data)
        active_path = save_active_config(best_config)
        print(f"\nActive configuration saved to: {active_path}")
    else:
        print(
            "\n[optimize] Validation did not meet quality gates; keeping existing config.json."
            f" (trades={val_trades} min={min_trades}, pnl={val_pnl:.2f}, pf={val_pf:.2f})"
        )

    if len(df_valid) > 0:
        print("\nGenerating chart for the validation period...")
        plot_backtest(df_val_ind, trades_val, f"{ticker}_validation")


if __name__ == "__main__":
    main()
