from __future__ import annotations

from typing import Dict, Any

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
  squeeze_cfg = data_cfg.get("squeeze", {}) or {}
  bb_period = int(squeeze_cfg.get("bb_period", 20))
  kurt_window = int(squeeze_cfg.get("kurtosis_window", 48))
  obv_window = int(squeeze_cfg.get("obv_window", 24))

  base = df.copy().sort_index()
  close = base["close"].astype(float)
  high = base["high"].astype(float)
  low = base["low"].astype(float)
  volume = base["volume"].astype(float)

  feats = pd.DataFrame(index=base.index)

  # Bollinger Bands
  ma = close.rolling(bb_period, min_periods=bb_period).mean()
  std = close.rolling(bb_period, min_periods=bb_period).std()
  upper = ma + 2 * std
  lower = ma - 2 * std
  width = (upper - lower) / (ma.abs() + 1e-9)
  feats["bb_width"] = width
  feats["bb_zscore"] = ((close - ma) / (std + 1e-9)).clip(-5, 5)

  # ATR e percentil aproximado (normalizado por preço)
  atr_period = int(squeeze_cfg.get("atr_period", 14))
  atr = _atr(base, atr_period)
  feats["atr"] = atr
  feats["atr_rel"] = atr / (close.abs() + 1e-9)

  # Retornos e kurtosis
  ret = close.pct_change()
  feats["ret_1"] = ret
  feats["ret_4"] = close.pct_change(4)
  feats["ret_12"] = close.pct_change(12)
  feats["ret_kurtosis"] = ret.rolling(kurt_window, min_periods=kurt_window).kurt()

  # OBV e inclinação
  direction = np.sign(close.diff().fillna(0.0))
  obv = (direction * volume).cumsum()
  feats["obv"] = obv
  feats["obv_slope"] = obv.diff(obv_window) / float(max(obv_window, 1))

  # Normalizações simples
  feats["log_close"] = np.log(close + 1e-9)
  feats["vol_zscore"] = (volume - volume.rolling(100).mean()) / (volume.rolling(100).std() + 1e-9)

  feats = feats.replace([np.inf, -np.inf], np.nan)
  feats = feats.dropna()
  return feats

