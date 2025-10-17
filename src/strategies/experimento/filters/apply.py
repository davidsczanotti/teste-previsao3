from __future__ import annotations

import pandas as pd

from .trend_mtf import apply_trend_gate
from .atr_threshold import apply_atr_threshold
from .volume import apply_volume_min


def apply_all_filters(df: pd.DataFrame, cfg_filters: dict) -> pd.DataFrame:
    out = df.copy()
    # Trend MTF (legacy) or generic MA trend
    if "ma_trend" in cfg_filters:
        tf = cfg_filters["ma_trend"].get("tf", "15m")
        fast_col = f"ma_fast_{tf}"
        slow_col = f"ma_slow_{tf}"
        col_exists = (fast_col in out.columns) and (slow_col in out.columns)
        if col_exists:
            out["trend_ok"] = (out[fast_col] > out[slow_col]).astype(int)
        else:
            out["trend_ok"] = 1
    elif "trend_tf" in cfg_filters:
        # Backward compatibility using precomputed ema_fast_15m/ema_slow_15m
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

    # VWAP bias (requires vwap_<tf>)
    if "vwap_bias" in cfg_filters:
        tf = cfg_filters["vwap_bias"].get("tf", "30m")
        mode = (cfg_filters["vwap_bias"].get("mode") or cfg_filters["vwap_bias"].get("bias") or "none").lower()
        vwap_col = f"vwap_{tf}"
        if vwap_col in out.columns:
            if mode in ("above", "long_only"):
                out["vwap_ok"] = (out["close"] >= out[vwap_col]).astype(int)
            elif mode in ("below", "short_only"):
                out["vwap_ok"] = (out["close"] <= out[vwap_col]).astype(int)
            else:
                out["vwap_ok"] = 1
        else:
            out["vwap_ok"] = 1
    else:
        out["vwap_ok"] = 1

    return out
