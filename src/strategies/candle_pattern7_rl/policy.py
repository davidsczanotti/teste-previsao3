from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float


def encode_last7(candles: List[Candle]) -> dict:
    """Extract simple features from the last 7 closed candles.

    Features include:
    - green_count / red_count
    - average true range proxy (mean of high-low)
    - simple moving average of closes (sma7)
    - last close and last candle direction
    """
    assert len(candles) == 7
    greens = sum(1 for c in candles if c.close > c.open)
    reds = 7 - greens
    ranges = [(c.high - c.low) for c in candles]
    sma7 = sum(c.close for c in candles) / 7.0
    last = candles[-1]
    return {
        "greens": greens,
        "reds": reds,
        "avg_range": sum(ranges) / 7.0 if ranges else 0.0,
        "sma7": sma7,
        "last_close": last.close,
        "last_dir_up": 1 if last.close >= last.open else 0,
    }


def default_signal_from_last7(
    candles: List[Candle],
    ma_short: Optional[float] = None,
    ma_long: Optional[float] = None,
    prev_ma_short: Optional[float] = None,
    prev_ma_long: Optional[float] = None,
) -> int:
    """Return -1 (sell), 0 (hold), 1 (buy) using pattern + moving averages.

    Heuristic (customisable):
      - Pattern bias from last 7 candles (greens/reds, SMA7).
      - Moving average confirmation via SMA7 (short) vs SMA40 (long).
      - Prefer fresh crossovers; otherwise require alignment (short above long).
    """
    f = encode_last7(candles)

    pattern_buy = f["greens"] >= 4 and f["last_close"] > f["sma7"]
    pattern_sell = f["reds"] >= 4 and f["last_close"] < f["sma7"]

    ma_ok = ma_short is not None and ma_long is not None
    prev_ok = prev_ma_short is not None and prev_ma_long is not None

    diff = (ma_short - ma_long) if ma_ok else 0.0
    prev_diff = (prev_ma_short - prev_ma_long) if (ma_ok and prev_ok) else diff

    cross_up = ma_ok and prev_ok and prev_diff <= 0.0 < diff
    cross_down = ma_ok and prev_ok and prev_diff >= 0.0 > diff

    trend_up = ma_ok and diff > 0.0
    trend_down = ma_ok and diff < 0.0

    if pattern_buy and (cross_up or trend_up):
        return 1
    if pattern_sell and (cross_down or trend_down):
        return -1
    return 0


def decide_action_from_signal(signal: int, position: int, allow_short: bool = True) -> int:
    """Map a directional signal to an action under single-position constraint.

    position: 0 flat, 1 long, -1 short
    return action:
      0 = hold
      1 = open long
      2 = close long
      3 = open short (if allowed)
      4 = close short

    Policy:
    - Flat: open if signal!=0 (respect allow_short)
    - Long: if signal<0 -> close long; else hold
    - Short: if signal>0 -> close short; else hold
    """
    if position == 0:
        if signal > 0:
            return 1
        if signal < 0 and allow_short:
            return 3
        return 0
    if position == 1:
        return 2 if signal < 0 else 0
    if position == -1:
        return 4 if signal > 0 else 0
    return 0
