#!/usr/bin/env python3
"""
Executes Walk-Forward Validation for the AL Brooks 1m strategy.

This script leverages the generic WalkForwardValidator to test the strategy's
robustness by periodically re-optimizing its parameters and validating them
on subsequent out-of-sample data.
"""

from ...utils.walk_forward import run_walk_forward_cli
from .backtest import backtest_al_brooks_inside_bar
from .optimize import make_objective


def main():
    """Main function to run the CLI for walk-forward validation."""
    run_walk_forward_cli(
        strategy_name="ALBROOKS",
        default_symbol="BTCUSDT",
        default_timeframe="1m",
        objective_func_creator=make_objective,
        backtest_func=backtest_al_brooks_inside_bar,
    )


if __name__ == "__main__":
    main()
