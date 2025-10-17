#!/usr/bin/env python3
"""
Generic Walk-Forward Validation Module
Implements a reusable walk-forward validation framework to test strategy robustness
across different market periods.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from .data_loader import load_data
from .metrics import calculate_metrics

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Class to implement walk-forward validation for any given strategy.
    """

    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        days: int,
        lot_size: float,
        min_trades_per_window: int,
        objective_func_creator: Callable,
        backtest_func: Callable,
        use_cache_only: bool = False,
    ):
        """
        Initializes the walk-forward validator.

        Args:
            strategy_name: Name of the strategy (e.g., "AL_BROOKS").
            symbol: Asset symbol (e.g., "BTCUSDT").
            timeframe: Candle timeframe (e.g., "1m", "15m").
            days: Total days of historical data to use.
            lot_size: Lot size for trading.
            min_trades_per_window: Minimum trades in validation to be considered a successful period.
            objective_func_creator: A function that takes (df, lot_size) and returns an Optuna objective function.
            backtest_func: The backtesting function for the strategy.
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.days = days
        self.lot_size = lot_size
        self.min_trades_per_window = min_trades_per_window
        self.objective_func_creator = objective_func_creator
        self.backtest_func = backtest_func
        self.results: List[Dict[str, Any]] = []
        self.summary_stats: Dict[str, Any] = {}
        self._aggregation_base: List[Dict[str, Any]] = []
        self.use_cache_only = bool(use_cache_only)

    def _get_candles_per_day(self) -> int:
        """Calculates the approximate number of candles per day for a given timeframe."""
        unit = self.timeframe[-1]
        try:
            value = int(self.timeframe[:-1])
        except ValueError:
            value = 1

        if unit == "m" and value > 0:
            return (24 * 60) // value
        if unit == "h" and value > 0:
            return 24 // value
        if unit == "d":
            return 1
        logger.warning(f"Could not parse timeframe '{self.timeframe}', defaulting to 24 candles/day.")
        return 24

    def create_periods(
        self, data: pd.DataFrame, optimization_window: int, validation_window: int, step_size: int
    ) -> List[Dict[str, Any]]:
        """Creates optimization and validation periods based on candle indices."""
        periods = []
        total_candles = len(data)
        candles_per_day = self._get_candles_per_day()

        opt_candles = optimization_window * candles_per_day
        val_candles = validation_window * candles_per_day
        step_candles = step_size * candles_per_day

        total_candles_needed = opt_candles + val_candles

        if total_candles_needed > total_candles:
            logger.warning(
                f"Insufficient data: {total_candles_needed} candles needed, but only {total_candles} available."
            )
            return periods

        current_start = 0
        while current_start + total_candles_needed <= total_candles:
            opt_end = current_start + opt_candles
            val_start = opt_end
            val_end = val_start + val_candles

            periods.append(
                {
                    "optimization_start": current_start,
                    "optimization_end": opt_end,
                    "validation_start": val_start,
                    "validation_end": val_end,
                    "period_number": len(periods) + 1,
                }
            )
            current_start += step_candles

        logger.info(f"Created {len(periods)} walk-forward periods.")
        return periods

    def run_single_period(self, data: pd.DataFrame, period: Dict[str, Any]) -> Dict[str, Any]:
        """Executes optimization and validation for a single period."""
        logger.info(
            f"Processing period {period['period_number']}: "
            f"Optimizing from candle {period['optimization_start']} to {period['optimization_end']}"
        )

        opt_data = data.iloc[period["optimization_start"] : period["optimization_end"]].reset_index(drop=True)

        try:
            study = optuna.create_study(direction="maximize")
            objective = self.objective_func_creator(opt_data, self.lot_size)
            study.optimize(objective, n_trials=50, show_progress_bar=False, gc_after_trial=True)
            best_params = study.best_params
            best_score = study.best_value

            logger.info(f"Best optimization score: {best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")

        except Exception as e:
            logger.error(f"Error during optimization for period {period['period_number']}: {e}", exc_info=True)
            return {"period": period, "optimization_success": False, "error": str(e)}

        val_data = data.iloc[period["validation_start"] : period["validation_end"]].reset_index(drop=True)

        try:
            trades, total_pnl, _ = self.backtest_func(val_data.copy(), lot_size=self.lot_size, **best_params)
            metrics = calculate_metrics(trades)

            result = {
                "period": period,
                "optimization_success": True,
                "validation_success": True,
                "best_params": best_params,
                "best_score": best_score,
                "total_pnl": total_pnl,
                "metrics": metrics,
                "validation_pnl": metrics.get("total_pnl", 0.0),
                "validation_trades": metrics.get("total_trades", 0),
                "validation_win_rate": metrics.get("win_rate", 0.0),
                "validation_profit_factor": metrics.get("profit_factor", 0.0),
                "validation_profit": metrics.get("total_profit", 0.0),
                "validation_loss": metrics.get("total_loss", 0.0),
                "meets_min_trades": metrics.get("total_trades", 0) >= self.min_trades_per_window,
            }

            logger.info(
                f"Validation result: P&L ${result['validation_pnl']:.2f}, "
                f"Trades: {result['validation_trades']}, "
                f"Win Rate {result['validation_win_rate']:.2%}, "
                f"Profit Factor {result['validation_profit_factor']:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error during validation for period {period['period_number']}: {e}", exc_info=True)
            return {
                "period": period,
                "optimization_success": True,
                "best_params": best_params,
                "best_score": best_score,
                "validation_success": False,
                "error": str(e),
            }

    def run_walk_forward(self, optimization_window: int, validation_window: int, step_size: int) -> Dict[str, Any]:
        """Executes the complete walk-forward validation."""
        logger.info("Starting walk-forward validation...")
        data = load_data(self.symbol, self.timeframe, self.days, use_cache_only=self.use_cache_only)
        logger.info(f"Data loaded: {len(data)} candles from {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}")

        periods = self.create_periods(data, optimization_window, validation_window, step_size)

        if not periods:
            logger.warning("No walk-forward periods could be created with the given parameters.")
            return {"results": [], "summary_stats": {}}

        for period in periods:
            result = self.run_single_period(data, period)
            self.results.append(result)

        self.calculate_summary_stats()
        report = self.generate_report()

        logger.info("Walk-forward validation finished!")
        return {"results": self.results, "summary_stats": self.summary_stats, "report": report}

    def calculate_summary_stats(self) -> None:
        """Calculates aggregated statistics from the validation results."""
        successful_periods = [r for r in self.results if r.get("optimization_success") and r.get("meets_min_trades")]
        periods_with_trades = [
            r for r in self.results if r.get("optimization_success") and r.get("validation_trades", 0) > 0
        ]

        if not periods_with_trades:
            logger.warning("No periods resulted in any trades during validation.")
            self.summary_stats = {"aggregation_scope": "none"}
            return

        if not successful_periods:
            logger.warning(
                f"No period met the minimum of {self.min_trades_per_window} trades. "
                f"Aggregating results from all {len(periods_with_trades)} periods that had trades."
            )
            self._aggregation_base = periods_with_trades
            aggregation_scope = "all_with_trades"
        else:
            self._aggregation_base = successful_periods
            aggregation_scope = "qualified"

        pnls = [r["validation_pnl"] for r in self._aggregation_base]
        win_rates = [r["validation_win_rate"] for r in self._aggregation_base]
        profit_factors = [r["validation_profit_factor"] for r in self._aggregation_base]

        total_profit = sum(r.get("validation_profit", 0.0) for r in self._aggregation_base)
        total_loss = sum(r.get("validation_loss", 0.0) for r in self._aggregation_base)

        aggregate_profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        self.summary_stats = {
            "total_periods": len(self.results),
            "successful_periods": len(successful_periods),
            "periods_with_trades": len(periods_with_trades),
            "success_rate": len(successful_periods) / len(self.results) if self.results else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls) if pnls else 0,
            "median_pnl": np.median(pnls) if pnls else 0,
            "std_pnl": np.std(pnls) if pnls else 0,
            "max_pnl": max(pnls) if pnls else 0,
            "min_pnl": min(pnls) if pnls else 0,
            "avg_win_rate": np.mean(win_rates) if win_rates else 0,
            "median_win_rate": np.median(win_rates) if win_rates else 0,
            "avg_profit_factor": np.mean(profit_factors) if profit_factors else 0,
            "median_profit_factor": np.median(profit_factors) if profit_factors else 0,
            "periods_with_profit": sum(1 for p in pnls if p > 0),
            "periods_with_loss": sum(1 for p in pnls if p < 0),
            "aggregate_profit_factor": aggregate_profit_factor,
            "min_trades_required": self.min_trades_per_window,
            "aggregation_scope": aggregation_scope,
        }

        logger.info("Aggregated Statistics:")
        logger.info(
            f"  - Successful Periods: {self.summary_stats['successful_periods']}/{self.summary_stats['total_periods']}"
        )
        logger.info(f"  - Total P&L: ${self.summary_stats['total_pnl']:.2f}")
        logger.info(f"  - Average P&L: ${self.summary_stats['avg_pnl']:.2f}")
        logger.info(f"  - Average Win Rate: {self.summary_stats['avg_win_rate']:.2%}")
        logger.info(f"  - Aggregate Profit Factor: {self.summary_stats['aggregate_profit_factor']:.2f}")

    def generate_report(self) -> Dict[str, Any]:
        """Generates a complete JSON report and a performance chart."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "summary_stats": self.summary_stats,
            "detailed_results": [],
        }

        for result in self.results:
            detailed_result = {
                "period_number": result["period"]["period_number"],
                "optimization_period": f"Candles {result['period']['optimization_start']} to {result['period']['optimization_end']}",
                "validation_period": f"Candles {result['period']['validation_start']} to {result['period']['validation_end']}",
                "optimization_success": result.get("optimization_success", False),
                "validation_success": result.get("validation_success", False),
                "best_score": result.get("best_score"),
                "validation_pnl": result.get("validation_pnl"),
                "validation_trades": result.get("validation_trades"),
                "validation_win_rate": result.get("validation_win_rate"),
                "validation_profit_factor": result.get("validation_profit_factor"),
                "meets_min_trades": result.get("meets_min_trades"),
                "best_params": result.get("best_params"),
                "error": result.get("error"),
            }
            report["detailed_results"].append(detailed_result)

        report_dir = "reports/walk_forward"
        os.makedirs(report_dir, exist_ok=True)
        report_path = f"{report_dir}/{self.strategy_name}_{self.symbol}_{self.timeframe}_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to: {report_path}")

        self.generate_performance_chart()

        return report

    def generate_performance_chart(self) -> None:
        """Generates and saves a chart of performance over the walk-forward periods."""
        if not self._aggregation_base:
            logger.warning("No results available to generate a performance chart.")
            return

        periods = [r["period"]["period_number"] for r in self._aggregation_base]
        pnls = [r["validation_pnl"] for r in self._aggregation_base]
        win_rates = [r["validation_win_rate"] for r in self._aggregation_base]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"Walk-Forward Performance: {self.strategy_name} on {self.symbol} ({self.timeframe})", fontsize=16)

        # P&L Chart
        colors = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]
        ax1.bar(periods, pnls, color=colors)
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax1.set_ylabel("P&L ($)")
        ax1.set_title("P&L per Validation Period")
        ax1.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
        ax1.set_xticks(periods)

        # Win Rate Chart
        ax2.plot(periods, win_rates, "o-", color="#1f77b4", label="Win Rate")
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="50% Mark")
        ax2.set_xlabel("Period Number")
        ax2.set_ylabel("Win Rate")
        ax2.set_title("Win Rate per Validation Period")
        ax2.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
        ax2.legend()
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        chart_dir = "reports/charts"
        os.makedirs(chart_dir, exist_ok=True)
        chart_path = f"{chart_dir}/walk_forward_{self.strategy_name}_{self.symbol}_{self.timeframe}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()

        logger.info(f"Performance chart saved to: {chart_path}")

    def print_summary_report(self) -> None:
        """Prints a formatted summary of the walk-forward validation results to the console."""
        summary = self.summary_stats
        if not summary:
            print("No summary statistics were generated.")
            return

        print("\n" + "=" * 60)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Strategy: {self.strategy_name}")
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Min. Trades per Window: {self.min_trades_per_window}")
        print("-" * 60)
        print(f"Total Periods Tested: {summary.get('total_periods', 0)}")
        print(f"Successful Periods (met min trades): {summary.get('successful_periods', 0)}")
        print(f"Periods with Any Trades: {summary.get('periods_with_trades', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.2%}")
        print("-" * 60)
        print(f"Total P&L (from aggregated periods): ${summary.get('total_pnl', 0):.2f}")
        print(f"Average P&L per Period: ${summary.get('avg_pnl', 0):.2f}")
        print(f"Average Win Rate: {summary.get('avg_win_rate', 0):.2%}")
        print(f"Average Profit Factor: {summary.get('avg_profit_factor', 0):.2f}")

        agg_pf = summary.get("aggregate_profit_factor", float("nan"))
        if np.isfinite(agg_pf):
            print(f"Aggregate Profit Factor: {agg_pf:.2f}")
        else:
            print("Aggregate Profit Factor: inf")

        scope = summary.get("aggregation_scope")
        if scope == "all_with_trades":
            print("\nNOTE: No window met the minimum trade requirement.")
            print("Metrics above are based on all periods that had at least one trade.")
        elif scope == "none":
            print("\nNOTE: No trades were executed in any validation window.")
        print("=" * 60)


def run_walk_forward_cli(
    strategy_name: str,
    default_symbol: str,
    default_timeframe: str,
    objective_func_creator: Callable,
    backtest_func: Callable,
):
    """
    Sets up and runs the CLI for walk-forward validation.

    Args:
        strategy_name: The name of the strategy.
        default_symbol: Default symbol for argparse.
        default_timeframe: Default timeframe for argparse.
        objective_func_creator: Function to create the Optuna objective.
        backtest_func: The backtesting function.
    """
    parser = argparse.ArgumentParser(description=f"Walk-Forward Validation for {strategy_name} Strategy")
    parser.add_argument("--symbol", type=str, default=default_symbol, help="Asset symbol.")
    parser.add_argument("--timeframe", type=str, default=default_timeframe, help="Candle timeframe.")
    parser.add_argument("--days", type=int, default=365, help="Total days of historical data.")
    parser.add_argument("--lot-size", type=float, default=0.1, help="Lot size for trades.")
    parser.add_argument("--opt-window", type=int, default=30, help="Optimization window size (days).")
    parser.add_argument("--val-window", type=int, default=15, help="Validation window size (days).")
    parser.add_argument("--step-size", type=int, default=15, help="Step size between periods (days).")
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum trades in validation window to consider a period successful.",
    )
    parser.add_argument("--cache-only", action="store_true", help="Use only local cache (no network)")
    args = parser.parse_args()

    validator = WalkForwardValidator(
        strategy_name=strategy_name,
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        lot_size=args.lot_size,
        min_trades_per_window=args.min_trades,
        objective_func_creator=objective_func_creator,
        backtest_func=backtest_func,
        use_cache_only=args.cache_only,
    )

    validator.run_walk_forward(
        optimization_window=args.opt_window,
        validation_window=args.val_window,
        step_size=args.step_size,
    )

    validator.print_summary_report()
