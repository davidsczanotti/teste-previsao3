from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


def _classify_bars(df: pd.DataFrame, body_min_pct: float, near_extreme_frac: float) -> None:
    rng = (df["high"] - df["low"]).astype(float)
    body = (df["close"] - df["open"]).abs().astype(float)
    body_pct = np.where(rng > 0, body / rng * 100.0, 0.0)
    df["bar_range"] = rng
    df["bar_body"] = body
    df["bar_body_pct"] = body_pct

    # Near extreme conditions (close near high/low for trend bars)
    near_high = (df["high"] - df["close"]).astype(float) <= (rng * float(near_extreme_frac))
    near_low = (df["close"] - df["low"]).astype(float) <= (rng * float(near_extreme_frac))

    df["bull_trend_bar"] = (df["close"] > df["open"]) & (body_pct >= body_min_pct) & near_high
    df["bear_trend_bar"] = (df["close"] < df["open"]) & (body_pct >= body_min_pct) & near_low
    df["doji_like"] = body_pct <= min(25.0, body_min_pct * 0.5)

    # Inside/outside bars
    df["inside_bar"] = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    df["outside_bar"] = (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))


def _compute_swings(df: pd.DataFrame, lookback: int) -> None:
    """Compute simple swing highs/lows using a symmetric lookback window.

    A pivot high at i is defined if high[i] is strictly the max within
    [i - lookback, i + lookback]. Similarly for pivot low.
    """
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    L = int(max(1, lookback))
    for i in range(n):
        a = max(0, i - L)
        b = min(n, i + L + 1)
        hmax = np.nanmax(highs[a:b]) if b > a else np.nan
        lmin = np.nanmin(lows[a:b]) if b > a else np.nan
        if np.isfinite(hmax) and highs[i] == hmax:
            sh[i] = True
        if np.isfinite(lmin) and lows[i] == lmin:
            sl[i] = True

    df["swing_high"] = sh
    df["swing_low"] = sl

    # Track the most recent swing high/low values for quick reference
    df["last_swing_high"] = np.where(sh, df["high"], np.nan)
    df["last_swing_low"] = np.where(sl, df["low"], np.nan)
    df["last_swing_high"] = df["last_swing_high"].ffill()
    df["last_swing_low"] = df["last_swing_low"].ffill()


def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    ema_fast = int(params.get("ema_fast_period", 20))
    ema_med = int(params.get("ema_medium_period", 50))
    ema_slow = int(params.get("ema_slow_period", 200))
    slope_lb = int(params.get("slope_lookback", 5))
    swing_lb = int(params.get("swing_lookback", 3))
    atr_len = int(params.get("atr_period", 14))
    body_min = float(params.get("bar_body_min_pct", 55.0))
    near_frac = float(params.get("near_extreme_frac", 0.25))

    df["ema_fast"] = ta.ema(df["close"], length=ema_fast)
    df["ema_medium"] = ta.ema(df["close"], length=ema_med)
    df["ema_slow"] = ta.ema(df["close"], length=ema_slow)

    # EMA20 slope (price units per slope_lookback bars)
    df["ema_fast_slope"] = df["ema_fast"] - df["ema_fast"].shift(slope_lb)

    # ATR for volatility and stop sizing
    df["atr"] = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=atr_len)

    _classify_bars(df, body_min, near_frac)
    _compute_swings(df, swing_lb)

    return df

