#!/usr/bin/env python3
"""
Walk-Forward Validation for the Al Brooks (book-style) strategy.
"""

from ...utils.walk_forward import run_walk_forward_cli
from .backtest import backtest_al_brooks_book
from .optimize import make_objective


def main():
    run_walk_forward_cli(
        strategy_name="ALBROOKS_BOOK",
        default_symbol="BTCUSDT",
        default_timeframe="1m",
        objective_func_creator=make_objective,
        backtest_func=backtest_al_brooks_book,
    )


if __name__ == "__main__":
    main()

