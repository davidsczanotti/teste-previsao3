from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(window=length, min_periods=1).mean()


# Additional MAs
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=1).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    # Linear weights 1..length
    if length <= 1:
        return series.copy()
    weights = np.arange(1, length + 1, dtype=float)
    def _calc(x: np.ndarray) -> float:
        return float(np.dot(x, weights[-len(x):]) / weights[-len(x):].sum())
    return series.rolling(window=length, min_periods=1).apply(_calc, raw=True)


def hma(series: pd.Series, length: int) -> pd.Series:
    # Hull MA: WMA( 2*WMA(n/2) - WMA(n), sqrt(n) )
    if length <= 1:
        return series.copy()
    n = int(length)
    wma_n = wma(series, n)
    wma_half = wma(series, max(1, n // 2))
    diff = 2 * wma_half - wma_n
    return wma(diff, max(1, int(np.sqrt(n))))


def vwap_daily(df: pd.DataFrame) -> pd.Series:
    # VWAP com reset diário (por data de close_time). Usa hlc3 como preço típico.
    price = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    vol = df["volume"].astype(float)
    # Agrupar por dia (UTC/naive conforme dados)
    day = pd.to_datetime(df["close_time"]).dt.date
    tpv = (price * vol).groupby(day).cumsum()
    vcv = vol.groupby(day).cumsum()
    return tpv / vcv.replace(0, np.nan)


def compute_ma(series: pd.Series, ma_type: str, length: int) -> pd.Series:
    t = (ma_type or "").strip().lower()
    if t == "ema":
        return ema(series, length)
    if t == "sma":
        return sma(series, length)
    if t == "wma":
        return wma(series, length)
    if t in ("hma", "hull"):
        return hma(series, length)
    # default fallback
    return ema(series, length)
