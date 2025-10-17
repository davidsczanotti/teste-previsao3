from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Signal:
    name: str
    direction: str  # "long" or "short"
    index: int      # signal bar index (uses last closed bar)
    entry: float
    stop: float
    reason: str
    meta: Optional[Dict] = None


def _is_bull_trend(df: pd.DataFrame, i: int, params: dict) -> bool:
    use_slope = bool(params.get("use_trend_slope", True))
    min_slope = float(params.get("min_ema_slope", 0.0))
    row = df.iloc[i]
    if np.isnan(row.get("ema_fast")):
        return False
    cond = (row["close"] >= row["ema_fast"]) and (row["ema_fast"] >= row.get("ema_medium", row["ema_fast"]))
    if use_slope:
        slope = row.get("ema_fast_slope")
        if np.isnan(slope):
            return False
        cond = cond and (slope >= min_slope)
    return bool(cond)


def _is_bear_trend(df: pd.DataFrame, i: int, params: dict) -> bool:
    use_slope = bool(params.get("use_trend_slope", True))
    min_slope = float(params.get("min_ema_slope", 0.0))
    row = df.iloc[i]
    if np.isnan(row.get("ema_fast")):
        return False
    cond = (row["close"] <= row["ema_fast"]) and (row["ema_fast"] <= row.get("ema_medium", row["ema_fast"]))
    if use_slope:
        slope = row.get("ema_fast_slope")
        if np.isnan(slope):
            return False
        cond = cond and (slope <= -min_slope)
    return bool(cond)


def _within_atr_distance(x: float, y: float, atr: float, max_mult: float) -> bool:
    if not np.isfinite(atr) or atr <= 0:
        return True  # if no ATR yet, don't reject by distance
    return abs(float(x) - float(y)) <= float(max_mult) * float(atr)


def detect_inside_trend(df: pd.DataFrame, i: int, params: dict) -> Optional[Signal]:
    """Inside-bar trend continuation setup.

    Long: last closed bar (i) is inside bar; bull trend context.
          Entry = high[i] + tick, Stop = low[i]. Short is symmetric.
    """
    if not bool(params.get("enable_inside_trend", True)):
        return None
    if i < 1:
        return None
    row = df.iloc[i]
    if not bool(row.get("inside_bar", False)):
        return None

    if _is_bull_trend(df, i, params):
        entry = float(row["high"])  # stop entry above the signal bar
        stop = float(row["low"])    # protective stop below signal bar
        return Signal(
            name="IB-Trend",
            direction="long",
            index=i,
            entry=entry,
            stop=stop,
            reason="Inside bar in bull trend",
        )
    if _is_bear_trend(df, i, params):
        entry = float(row["low"])   # stop entry below the signal bar
        stop = float(row["high"])   # protective stop above signal bar
        return Signal(
            name="IB-Trend",
            direction="short",
            index=i,
            entry=entry,
            stop=stop,
            reason="Inside bar in bear trend",
        )
    return None


def detect_h2_l2(df: pd.DataFrame, i: int, params: dict) -> Optional[Signal]:
    """Approximate H2/L2 in-trend entries.

    Bull H2: in bull trend, count two down attempts (bars making lower lows)
    in the recent window; if the signal bar is bullish, buy stop above it.
    Bear L2: symmetric.
    """
    if not bool(params.get("enable_h2_l2", True)):
        return None
    if i < 3:
        return None
    lb = int(max(3, params.get("bo_lookback", 20)))

    # Count down attempts in recent window
    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)

    start = max(1, i - lb)
    down_attempts = 0
    up_attempts = 0
    for k in range(start + 1, i + 1):
        if lows[k] < lows[k - 1]:
            down_attempts += 1
        if highs[k] > highs[k - 1]:
            up_attempts += 1

    row = df.iloc[i]
    # Bullish H2
    if _is_bull_trend(df, i, params) and down_attempts >= 2 and closes[i] > opens[i]:
        return Signal(
            name="H2",
            direction="long",
            index=i,
            entry=float(row["high"]),
            stop=float(row["low"]),
            reason=f"H2 after {down_attempts} down attempts in bull trend",
        )
    # Bearish L2
    if _is_bear_trend(df, i, params) and up_attempts >= 2 and closes[i] < opens[i]:
        return Signal(
            name="L2",
            direction="short",
            index=i,
            entry=float(row["low"]),
            stop=float(row["high"]),
            reason=f"L2 after {up_attempts} up attempts in bear trend",
        )
    return None


def detect_bo_pb(df: pd.DataFrame, i: int, params: dict) -> Optional[Signal]:
    """Breakout + Pullback entry approximation.

    Bull: a recent breakout bar above last swing high followed by a pullback
          near EMA20; entry above the pullback bar high; stop below its low.
    Bear: symmetric below swing low.
    """
    if not bool(params.get("enable_bo_pb", True)):
        return None
    if i < 5:
        return None

    lb = int(params.get("bo_lookback", 20))
    max_dist = float(params.get("max_ema_distance_atr", 1.0))
    atr = float(df.iloc[i].get("atr", np.nan))

    # Find a recent breakout bar in the lookback window
    start = max(1, i - lb)
    last_sh = float(df.iloc[i].get("last_swing_high", np.nan))
    last_sl = float(df.iloc[i].get("last_swing_low", np.nan))

    breakout_up = False
    breakout_dn = False
    for k in range(start, i + 1):
        r = df.iloc[k]
        if bool(r.get("bull_trend_bar", False)) and np.isfinite(last_sh) and (r["high"] > last_sh):
            breakout_up = True
        if bool(r.get("bear_trend_bar", False)) and np.isfinite(last_sl) and (r["low"] < last_sl):
            breakout_dn = True

    row = df.iloc[i]
    ema = float(row.get("ema_fast", np.nan))
    if breakout_up and _is_bull_trend(df, i, params):
        # Pullback proximity to EMA20
        if _within_atr_distance(row["low"], ema, atr, max_dist):
            return Signal(
                name="BO-PB",
                direction="long",
                index=i,
                entry=float(row["high"]),
                stop=float(row["low"]),
                reason="Breakout above swing high + PB near EMA20",
            )
    if breakout_dn and _is_bear_trend(df, i, params):
        if _within_atr_distance(row["high"], ema, atr, max_dist):
            return Signal(
                name="BO-PB",
                direction="short",
                index=i,
                entry=float(row["low"]),
                stop=float(row["high"]),
                reason="Breakout below swing low + PB near EMA20",
            )
    return None


def detect_signals(df: pd.DataFrame, i: int, params: dict) -> List[Signal]:
    """Returns all candidate signals at bar index i (last closed bar)."""
    sigs: List[Signal] = []
    for fn in (detect_bo_pb, detect_h2_l2, detect_inside_trend):
        s = fn(df, i, params)
        if s is not None:
            sigs.append(s)
    return sigs

