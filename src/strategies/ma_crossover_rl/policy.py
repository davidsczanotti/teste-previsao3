from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MASnapshot:
    ma_short: float
    ma_mid: float
    ma_long: float
    ma_short_prev: float
    ma_mid_prev: float
    ma_long_prev: float


def ma_signal(snapshot: MASnapshot) -> int:
    """Return -1 (sell), 0 (hold), 1 (buy) using MA(7)/MA(40) context.

    Logic:
      - Fresh crossover up (short crosses above long) → buy.
      - Fresh crossover down → sell.
      - Otherwise require separation + slope confirmation:
          * Buy if short > long with positive slope.
          * Sell if short < long with negative slope.
    """
    ma_s = snapshot.ma_short
    ma_l = snapshot.ma_long
    ma_s_prev = snapshot.ma_short_prev
    ma_l_prev = snapshot.ma_long_prev

    ma_m = snapshot.ma_mid
    ma_m_prev = snapshot.ma_mid_prev

    diff_s_m = ma_s - ma_m
    diff_m_l = ma_m - ma_l
    diff_s_l = ma_s - ma_l

    prev_diff_s_m = ma_s_prev - ma_m_prev
    prev_diff_m_l = ma_m_prev - ma_l_prev
    prev_diff_s_l = ma_s_prev - ma_l_prev

    cross_up = (prev_diff_s_m <= 0.0 and diff_s_m > 0.0) or (prev_diff_m_l <= 0.0 and diff_m_l > 0.0)
    cross_down = (prev_diff_s_m >= 0.0 and diff_s_m < 0.0) or (prev_diff_m_l >= 0.0 and diff_m_l < 0.0)

    slope_short = ma_s - ma_s_prev
    slope_mid = ma_m - ma_m_prev
    slope_long = ma_l - ma_l_prev

    if cross_up:
        return 1
    if cross_down:
        return -1
    if diff_s_l > 0.0 and diff_s_m > 0.0 and diff_m_l > 0.0 and slope_short >= slope_mid >= slope_long >= 0.0:
        return 1
    if diff_s_l < 0.0 and diff_s_m < 0.0 and diff_m_l < 0.0 and slope_short <= slope_mid <= slope_long <= 0.0:
        return -1
    return 0


def decide_action_from_signal(signal: int, position: int, allow_short: bool = True) -> int:
    """Map MA signal to discrete action under single-position constraint."""
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
