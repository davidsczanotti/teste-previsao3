from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class EmaOnlyParams:
    ema_period: int = 8
    lot_size: float = 0.001  # BTC quantity for trades
    fee_rate: float = 0.001  # 0.1% per side
    use_cross: bool = False  # if True, require crossing events instead of simple above/below


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def backtest_ema_only(
    df: pd.DataFrame,
    params: EmaOnlyParams = EmaOnlyParams(),
    initial_capital: float = 1_000.0,
) -> Tuple[List[Dict], float, Dict[str, float]]:
    """
    Very simple EMA-based strategy:
      - Long-only.
      - Enter when close < EMA (mean reversion style) OR when a down-cross occurs (if use_cross).
      - Exit when close > EMA OR when an up-cross occurs (if use_cross).

    PnL is computed in quote currency using a fixed lot size of base asset.
    """
    if df.empty:
        raise ValueError("DataFrame vazio para backtest_ema_only")

    df = df.sort_values("Date").reset_index(drop=True).copy()
    closes = df["close"].astype(float)
    ema = compute_ema(closes, params.ema_period)

    # Start after EMA warms up
    start = max(params.ema_period + 1, 2)

    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    realized_pnl = 0.0
    equity_curve = [initial_capital]
    trades: List[Dict] = []

    for i in range(start, len(df)):
        p_prev = float(closes.iloc[i - 1])
        e_prev = float(ema.iloc[i - 1]) if not np.isnan(ema.iloc[i - 1]) else np.nan
        p = float(closes.iloc[i])
        e = float(ema.iloc[i]) if not np.isnan(ema.iloc[i]) else np.nan
        t = df["Date"].iloc[i]

        if np.isnan(e) or np.isnan(e_prev):
            equity_curve.append(initial_capital + realized_pnl)
            continue

        # Entry/exit conditions
        if position == 0:
            if params.use_cross:
                enter = (p_prev >= e_prev) and (p < e)
            else:
                enter = p < e
            if enter:
                position = 1
                entry_price = p
                trades.append({"date": t, "action": "BUY", "price": p})
                realized_pnl -= params.fee_rate * p * params.lot_size

        else:  # position == 1
            if params.use_cross:
                exit_ = (p_prev <= e_prev) and (p > e)
            else:
                exit_ = p > e
            if exit_:
                pnl = (p - entry_price) * params.lot_size
                realized_pnl += pnl
                trades.append({"date": t, "action": "SELL", "price": p, "pnl": pnl})
                realized_pnl -= params.fee_rate * p * params.lot_size
                position = 0
                entry_price = 0.0

        # Mark-to-market equity
        if position == 1:
            unreal = (p - entry_price) * params.lot_size
        else:
            unreal = 0.0
        equity_curve.append(initial_capital + realized_pnl + unreal)

    # Close on last bar if still open (at close price)
    if position == 1:
        p = float(closes.iloc[-1])
        t = df["Date"].iloc[-1]
        pnl = (p - entry_price) * params.lot_size
        realized_pnl += pnl
        trades.append({"date": t, "action": "SELL (final)", "price": p, "pnl": pnl})
        realized_pnl -= params.fee_rate * p * params.lot_size
        equity_curve[-1] = initial_capital + realized_pnl

    # Metrics
    total_pnl = float(realized_pnl)
    closed = [tr for tr in trades if "pnl" in tr]
    n_trades = len(closed)
    wins = len([tr for tr in closed if tr["pnl"] > 0])
    win_rate = (wins / n_trades * 100.0) if n_trades else 0.0
    ret_pct = (total_pnl / initial_capital * 100.0) if initial_capital else 0.0
    running_max = np.maximum.accumulate(np.array(equity_curve, dtype=float))
    dd = (np.array(equity_curve) - running_max) / running_max * 100.0
    max_dd_pct = float(dd.min()) if len(dd) else 0.0
    avg_pnl = (total_pnl / n_trades) if n_trades else 0.0

    stats = {
        "pnl": total_pnl,
        "num_trades": n_trades,
        "win_rate": win_rate,
        "return_pct": ret_pct,
        "avg_pnl_per_trade": avg_pnl,
        "max_drawdown_pct": max_dd_pct,
        "fee_rate": params.fee_rate,
        "ema_period": params.ema_period,
        "use_cross": params.use_cross,
        "lot_size": params.lot_size,
    }

    return trades, total_pnl, stats

