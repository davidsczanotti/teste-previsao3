from __future__ import annotations

import pandas as pd


def initial_stop_atr(entry_price: float, atr_value: float, mult: float, side: str = "long") -> float:
    if side == "long":
        return entry_price - mult * atr_value
    else:
        return entry_price + mult * atr_value


def update_trailing_atr(current_trailing: float, close: float, atr_value: float, mult: float, side: str = "long") -> float:
    if side == "long":
        # Raise trailing stop if close - mult*atr is higher
        candidate = close - mult * atr_value
        return max(current_trailing, candidate) if current_trailing is not None else candidate
    else:
        candidate = close + mult * atr_value
        return min(current_trailing, candidate) if current_trailing is not None else candidate

