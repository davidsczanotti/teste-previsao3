from __future__ import annotations

import pandas as pd


def apply_atr_threshold(df: pd.DataFrame, atr_col: str, min_frac: float) -> pd.Series:
    """ATR must be above a fraction of current price (proxy for volatility)."""
    return (df[atr_col] >= (df["close"] * min_frac)).astype(int)

