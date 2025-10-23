from __future__ import annotations

from datetime import datetime, timedelta, UTC
from typing import Optional

import pandas as pd

from ...utils.data_loader import load_data
from .features import compute_features


def load_btc_1h(days: int = 3650, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    if start and end:
        df = load_data_range("BTCUSDT", "1h", start, end, use_cache_only=True)
    else:
        df = load_data("BTCUSDT", "1h", days=days, use_cache_only=True)
    return df


def load_data_range(symbol: str, timeframe: str, start_date: str, end_date: str, use_cache_only: bool = True) -> pd.DataFrame:
    from ...utils.data_loader import load_data_range as base_loader

    return base_loader(symbol, timeframe, start_date, end_date, use_cache_only=use_cache_only)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df.index = pd.to_datetime(df["Date"])
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column")
    feats = compute_features(df)
    df_aligned = df.loc[feats.index]
    dataset = df_aligned.copy()
    dataset[feats.columns] = feats
    if "Date" in dataset.columns:
        dataset = dataset.drop(columns=["Date"])
    return dataset
