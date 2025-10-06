"""7-candle pattern strategy scaffold.

This package provides:
- A simple, pluggable policy that inspects the last 7 closed candles
  and emits a directional signal (buy/sell/hold).
- A minimal backtester that respects single-position constraints, fees,
  and basic statistics.

Run the backtest:
  poetry run python -m src.strategies.candle_pattern7.backtest \
    --ticker BTCUSDT --interval 15m --days 90 --lot_size 0.001 --fee_rate 0.001
"""

