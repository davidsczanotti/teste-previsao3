import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategies.ema_only.backtest import EmaOnlyParams, backtest_ema_only, compute_ema


def test_ema_cross_signals_follow_fast_slow_cross():
    """
    Garante que o modo ema_cross abre no cruzamento fast>slow e fecha no cruzamento fast<slow.
    Usa uma série simples onde sabemos exatamente onde o cruzamento acontece.
    """
    prices = [100, 99, 98, 97, 98, 99, 100, 101, 100, 99]
    dates = pd.date_range("2025-01-01", periods=len(prices), freq="h")
    df = pd.DataFrame({"Date": dates, "close": prices})

    params = EmaOnlyParams(
        ema_period=2,
        slow_ema_period=3,
        signal_mode="ema_cross",
        lot_size=1.0,
        fee_rate=0.0,
    )

    trades, _, _ = backtest_ema_only(df, params=params, initial_capital=1000.0)

    # Recalcula fast/slow e identifica as barras de cruzamento com a mesma lógica do backtest.
    fast = compute_ema(df["close"], params.ema_period)
    slow = compute_ema(df["close"], params.slow_ema_period)
    start = max(
        params.ema_period + 1,
        params.slow_ema_period + 1,
        params.trend_filter_period + 1 if params.trend_filter_period else 0,
        2,
    )

    def _find_cross(direction: str) -> int | None:
        for i in range(start, len(df)):
            e_prev, e = fast.iloc[i - 1], fast.iloc[i]
            s_prev, s = slow.iloc[i - 1], slow.iloc[i]
            cross_up = (e_prev <= s_prev) and (e > s)
            cross_down = (e_prev >= s_prev) and (e < s)
            if direction == "up" and cross_up:
                return i
            if direction == "down" and cross_down:
                return i
        return None

    idx_up = _find_cross("up")
    idx_down = _find_cross("down")

    assert idx_up is not None and idx_down is not None, "Esperávamos um cruzamento de alta e um de baixa."
    assert len(trades) == 2, "Deve haver exatamente um BUY e um SELL."
    assert trades[0]["action"] == "BUY"
    assert trades[0]["date"] == df["Date"].iloc[idx_up]
    assert trades[1]["action"] == "SELL"
    assert trades[1]["date"] == df["Date"].iloc[idx_down]
