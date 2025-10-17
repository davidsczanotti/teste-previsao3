from __future__ import annotations

import pandas as pd


def apply_volume_min(df: pd.DataFrame, percentile: float) -> pd.Series:
    """Allow bars with volume above a given percentile of rolling distribution."""
    p = df["volume"].rolling(window=200, min_periods=20).quantile(percentile)
    return (df["volume"] >= p.fillna(0)).astype(int)

