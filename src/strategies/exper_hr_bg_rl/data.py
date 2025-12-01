from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from ...utils.data_loader import load_data, load_data_range
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
            raise ValueError("Se usar data.start/data.end, ambos devem ser definidos.")
        df = load_data_range(symbol, timeframe, start, end, use_cache_only=True)
    else:
        df = load_data(symbol, timeframe, days=days, use_cache_only=True)
    if df.empty:
        raise ValueError(f"Nenhum dado disponível para {symbol} @ {timeframe}. Atualize o cache local.")
    return _ensure_datetime_index(df)


def build_dataset(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp]]:
    raw = load_ohlcv(config)
    feats = compute_features(raw, config=config)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()

    aligned = raw.loc[feats.index].copy()
    price_cols = ["open", "high", "low", "close", "volume"]
    for col in price_cols:
        if col not in aligned.columns:
            raise ValueError(f"Coluna obrigatória ausente em OHLCV: {col}")

    price_df = aligned[price_cols].reset_index(drop=True)
    feat_df = feats.reset_index(drop=True)
    timestamps = list(feats.index)
    if len(price_df) != len(feat_df):
        raise ValueError("price_df e feat_df desalinhados em comprimento.")
    return price_df, feat_df, timestamps

