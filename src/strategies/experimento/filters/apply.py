from __future__ import annotations

import pandas as pd

from .trend_mtf import apply_trend_gate
from .atr_threshold import apply_atr_threshold
from .volume import apply_volume_min


def apply_all_filters(df: pd.DataFrame, cfg_filters: dict) -> pd.DataFrame:
    out = df.copy()
    # Trend MTF (requires ema_fast_15m, ema_slow_15m already merged)
    if "trend_tf" in cfg_filters:
        out["trend_ok"] = apply_trend_gate(out, "ema_fast_15m", "ema_slow_15m").fillna(0).astype(int)
    else:
        out["trend_ok"] = 1

    # ATR min threshold (requires atr_30m)
    if "atr_min" in cfg_filters:
        out["atr_ok"] = apply_atr_threshold(out, "atr_30m", cfg_filters["atr_min"]["min_atr_frac"]).astype(int)
    else:
        out["atr_ok"] = 1

    # Volume percentile
    if "volume_min" in cfg_filters:
        out["vol_ok"] = apply_volume_min(out, cfg_filters["volume_min"]["percentile"]).astype(int)
    else:
        out["vol_ok"] = 1

    return out

