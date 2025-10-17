from __future__ import annotations

import pandas as pd


def apply_trend_gate(df: pd.DataFrame, ema_fast_col: str, ema_slow_col: str) -> pd.Series:
    """Trend gate based on context TF: allow only if ema_fast_ctx > ema_slow_ctx."""
    return (df[ema_fast_col] > df[ema_slow_col]).astype(int)

