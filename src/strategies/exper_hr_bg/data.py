from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from ...utils.data_loader import load_data_range, load_data
from .features import compute_features


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
  out = df.copy()
  if not isinstance(out.index, pd.DatetimeIndex):
    if "Date" not in out.columns:
      raise ValueError("DataFrame precisa de DatetimeIndex ou coluna 'Date'.")
    out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce")
    out = out.dropna(subset=["Date"]).set_index("Date")
  else:
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
  return out.sort_index()


def load_ohlcv(config: Dict[str, Any]) -> pd.DataFrame:
  data_cfg = config.get("data", {})
  symbol = data_cfg.get("base_symbol", "BTCUSDT")
  timeframe = data_cfg.get("timeframe", "1h")
  days = int(data_cfg.get("lookback_days", 3650))
  start = data_cfg.get("start")
  end = data_cfg.get("end")

  if start or end:
    if start is None or end is None:
      raise ValueError("Se usar start/end em data, ambos devem ser definidos.")
    df = load_data_range(symbol, timeframe, start, end, use_cache_only=True)
  else:
    df = load_data(symbol, timeframe, days=days, use_cache_only=True)
  if df.empty:
    raise ValueError(f"Nenhum dado disponível para {symbol} @ {timeframe}. Atualize o cache local.")
  return _ensure_datetime_index(df)


@dataclass
class Dataset:
  features: np.ndarray
  labels: np.ndarray
  close: np.ndarray
  index: pd.DatetimeIndex


def build_dataset(config: Dict[str, Any]) -> Dataset:
  data_cfg = config.get("data", {})
  label_cfg = data_cfg.get("label", {}) or {}
  horizon = int(label_cfg.get("horizon_bars", 6))
  up_thr = float(label_cfg.get("up_return_threshold", 0.003))
  down_thr = float(label_cfg.get("down_return_threshold", 0.003))

  raw = load_ohlcv(config)
  feats = compute_features(raw, config=config)
  feats = feats.replace([np.inf, -np.inf], np.nan).dropna()

  aligned = raw.loc[feats.index].copy()
  close = aligned["close"].astype(float)

  fwd_ret = (close.shift(-horizon) / close) - 1.0
  valid = fwd_ret.notna()
  if not valid.any():
    raise ValueError("Nenhum retorno futuro válido encontrado para rotular.")

  fwd = fwd_ret[valid]
  labels = np.zeros(len(fwd), dtype=np.int64)
  labels[fwd > up_thr] = 1
  labels[fwd < -down_thr] = 2

  X = feats.loc[valid].to_numpy(dtype=np.float32)
  idx = feats.loc[valid].index
  close_arr = close.loc[valid].to_numpy(dtype=np.float32)
  return Dataset(features=X, labels=labels, close=close_arr, index=idx)

