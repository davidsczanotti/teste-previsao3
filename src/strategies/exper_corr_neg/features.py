from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()
    return (series - mean) / (std + 1e-9)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute feature set for the mixture-of-experts agent.

    The input ``df`` must contain columns ``open, high, low, close, volume`` and an index Date.
    """

    out = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Trend features
    out["ema_fast"] = _ema(close, 12)
    out["ema_slow"] = _ema(close, 26)
    out["ema_cross"] = out["ema_fast"] - out["ema_slow"]
    out["donchian_up"] = high.rolling(20, min_periods=20).max()
    out["donchian_down"] = low.rolling(20, min_periods=20).min()
    out["donchian_mid"] = (out["donchian_up"] + out["donchian_down"]) / 2

    # Mean reversion features
    out["rsi_2"] = _rsi(close, 2)
    out["rsi_14"] = _rsi(close, 14)
    rolling_mean = close.rolling(20, min_periods=20).mean()
    rolling_std = close.rolling(20, min_periods=20).std()
    out["bb_upper"] = rolling_mean + 2 * rolling_std
    out["bb_lower"] = rolling_mean - 2 * rolling_std
    out["bb_width"] = out["bb_upper"] - out["bb_lower"]
    out["close_zscore"] = _zscore(close, 20)

    # Volatility features
    out["atr_14"] = _atr(df, 14)
    returns = close.pct_change()
    out["rv_10"] = returns.rolling(10, min_periods=10).std() * np.sqrt(10)
    out["vol_percentile"] = out["atr_14"].rolling(200).rank(pct=True)

    # Volume / flow features
    obv = volume.where(close > close.shift(), -volume.where(close < close.shift(), 0.0)).fillna(0.0)
    out["obv"] = obv.cumsum()
    mf_mult = ((close - low) - (high - close)) / (high - low + 1e-9)
    mf_vol = mf_mult * volume
    out["chaikin"] = mf_vol.rolling(10, min_periods=10).sum()
    out["volume_zscore"] = _zscore(volume, 20)

    # Squeeze / breakout features
    out["bb_ratio"] = out["bb_width"] / (rolling_mean + 1e-9)
    out["atr_norm"] = out["atr_14"] / (out["donchian_mid"].abs() + 1e-9)
    out["squeeze"] = out["bb_width"] / (out["atr_14"] + 1e-9)

    # Context features
    out["ret_1"] = returns
    out["ret_5"] = close.pct_change(5)
    out["log_volume"] = np.log1p(volume)
    out["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    return out.dropna()

