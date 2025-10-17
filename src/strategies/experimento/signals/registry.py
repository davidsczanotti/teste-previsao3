from __future__ import annotations

from typing import Callable, Dict
import pandas as pd

from .ema_cross import generate_signals as _ema_cross


SignalFn = Callable[[pd.DataFrame, dict], pd.DataFrame]

_REGISTRY: Dict[str, SignalFn] = {}


def register(name: str, fn: SignalFn) -> None:
    _REGISTRY[name] = fn


def get(name: str) -> SignalFn:
    return _REGISTRY[name]


def generate(df: pd.DataFrame, name: str, params: dict) -> pd.DataFrame:
    return get(name)(df, **params)


# Builtin registrations
def _sig_ema_cross(df: pd.DataFrame, **params):
    side = params.get("side", "long")
    exit_on_cross = bool(params.get("exit_on_cross", False))
    fast_col = params.get("fast_col", "ema_fast_30m")
    slow_col = params.get("slow_col", "ema_slow_30m")
    return _ema_cross(df, fast_col=fast_col, slow_col=slow_col, side=side, exit_on_cross=exit_on_cross)


register("ema_cross", _sig_ema_cross)

