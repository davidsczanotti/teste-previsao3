from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift()
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def compute_features(df: pd.DataFrame, *, config: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    feat_cfg = data_cfg.get("features", {}) or {}
    atr_period = int(feat_cfg.get("atr_period", 14))
    rv_windows: List[int] = feat_cfg.get("realized_vol_windows", [24, 72]) or [24, 72]
    range_windows: List[int] = feat_cfg.get("range_windows", [14, 48]) or [14, 48]

    base = df.copy().sort_index()
    open_ = base["open"].astype(float)
    high = base["high"].astype(float)
    low = base["low"].astype(float)
    close = base["close"].astype(float)
    volume = base["volume"].astype(float)

    feats = pd.DataFrame(index=base.index)

    # Range-based volatility
    range_hl = (high - low) / (close.abs() + 1e-9)
    feats["range_hl"] = range_hl

    for w in range_windows:
        feats[f"range_hl_mean_{w}"] = range_hl.rolling(w, min_periods=w).mean()
        feats[f"range_hl_std_{w}"] = range_hl.rolling(w, min_periods=w).std()

    # ATR
    atr = _atr(base, atr_period)
    feats["atr"] = atr
    feats["atr_rel"] = atr / (close.abs() + 1e-9)

    # Log returns
    log_close = np.log(close + 1e-9)
    ret1 = log_close.diff()
    feats["ret_1"] = ret1
    feats["ret_4"] = log_close.diff(4)
    feats["ret_12"] = log_close.diff(12)

    # Realized volatility windows
    for w in rv_windows:
        rv = ret1.rolling(w, min_periods=w).std()
        feats[f"rv_{w}"] = rv
        feats[f"rv_change_{w}"] = rv.diff()

    # Parkinson volatility (range-based)
    parkinson = (1.0 / (4.0 * np.log(2.0))) * (np.log((high + 1e-9) / (low + 1e-9)) ** 2)
    feats["parkinson"] = parkinson.rolling(atr_period, min_periods=atr_period).mean()

    # Volume features
    vol_ma = volume.rolling(48, min_periods=24).mean()
    vol_std = volume.rolling(48, min_periods=24).std()
    feats["vol_zscore"] = (volume - vol_ma) / (vol_std + 1e-9)

    # Kurtosis/skew de retornos
    feats["ret_kurtosis_48"] = ret1.rolling(48, min_periods=48).kurt()
    feats["ret_skew_48"] = ret1.rolling(48, min_periods=48).skew()

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.dropna()
    return feats

