"""MA(7) x MA(40) crossover strategy (RL + backtest).

Run the backtest:
  poetry run python -m src.strategies.ma_crossover_rl.backtest \
    --ticker BTCUSDT --interval 15m --days 365 --lot_size 0.001 --fee_rate 0.001
"""

