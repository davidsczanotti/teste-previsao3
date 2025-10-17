from __future__ import annotations

from typing import Callable, Dict
import pandas as pd

from .common import ema, atr


IndicatorFn = Callable[[pd.DataFrame, dict], pd.DataFrame]

_REGISTRY: Dict[str, IndicatorFn] = {}


def register(name: str, fn: IndicatorFn) -> None:
    _REGISTRY[name] = fn


def get(name: str) -> IndicatorFn:
    return _REGISTRY[name]


def apply_indicator(df: pd.DataFrame, name: str, params: dict) -> pd.DataFrame:
    return get(name)(df, params)


# Builtins
def _ind_ema(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    fast = int(params.get("fast", 9))
    slow = int(params.get("slow", 21))
    out = df.copy()
    out[f"ema_fast"] = ema(out["close"], fast)
    out[f"ema_slow"] = ema(out["close"], slow)
    return out


def _ind_atr(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    length = int(params.get("length", 14))
    out = df.copy()
    out[f"atr_{length}"] = atr(out, length)
    return out


register("ema", _ind_ema)
register("atr", _ind_atr)

