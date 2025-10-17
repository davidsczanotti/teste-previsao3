from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..risk.sizing import fixed_fraction
from ..risk.stops import initial_stop_atr, update_trailing_atr
from ..risk.costs import fee_amount, apply_slippage


@dataclass
class BacktestConfig:
    capital: float
    fee_bp: float
    slippage_ticks: float
    tick_size: float
    stop_mult: float
    trailing_mult: float
    side: str = "long"  # "long", "short", or "both"
    exit_on_cross: bool = False


def backtest_ema_cross(
    df: pd.DataFrame,
    run_id: str,
    cfg: BacktestConfig,
) -> Tuple[List[Dict], float, pd.DataFrame]:
    """
    Minimal backtest: long-only EMA cross with ATR stop + trailing ATR, costs and slippage.
    df requires columns: close_time, open/high/low/close/volume, ema_fast_30m, ema_slow_30m, atr_30m,
    optional filters: trend_ok, atr_ok, vol_ok, and signal column (1 for buy).
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    capital = cfg.capital
    position = None  # dict with keys: side, qty, entry_price, stop, entry_idx, entry_fee
    trades: List[Dict] = []

    for i in range(n):
        row = df.iloc[i]
        time = pd.to_datetime(row["close_time"]).to_pydatetime()
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr_val = float(row.get("atr_30m", np.nan))

        # Update trailing stop if in position
        if position is not None:
            # Trailing stop update (optional)
            if cfg.trailing_mult and cfg.trailing_mult > 0 and position.get("stop") is not None:
                trail = update_trailing_atr(position["stop"], price, atr_val, cfg.trailing_mult, side=position["side"])
                if position["side"] == "long":
                    position["stop"] = max(position["stop"], trail)
                else:
                    position["stop"] = min(position["stop"], trail)

            # Check stop hit within bar (only if stop set)
            if position.get("stop") is not None:
                if position["side"] == "long" and low <= position["stop"]:
                    exit_price = apply_slippage(position["stop"], side="sell", slippage_ticks=cfg.slippage_ticks, tick_size=cfg.tick_size)
                    notional = position["qty"] * exit_price
                    fee = fee_amount(notional, cfg.fee_bp)
                    pnl = position["qty"] * (exit_price - position["entry_price"]) - position["entry_fee"] - fee
                    capital += pnl
                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "entry_time": df.loc[position["entry_idx"], "close_time"],
                        "exit_time": row["close_time"],
                        "side": position["side"],
                        "qty": position["qty"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                    })
                    position = None
                    continue
                if position["side"] == "short" and high >= position["stop"]:
                    exit_price = apply_slippage(position["stop"], side="buy", slippage_ticks=cfg.slippage_ticks, tick_size=cfg.tick_size)
                    notional = position["qty"] * exit_price
                    fee = fee_amount(notional, cfg.fee_bp)
                    pnl = position["qty"] * (position["entry_price"] - exit_price) - position["entry_fee"] - fee
                    capital += pnl
                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "entry_time": df.loc[position["entry_idx"], "close_time"],
                        "exit_time": row["close_time"],
                        "side": position["side"],
                        "qty": position["qty"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                    })
                    position = None
                    continue

            # Optional: exit on cross reversal at close
            if cfg.exit_on_cross and int(row.get("signal", 0)) != 0:
                sig = int(row.get("signal", 0))
                if position["side"] == "long" and sig < 0:
                    exit_price = apply_slippage(price, side="sell", slippage_ticks=cfg.slippage_ticks, tick_size=cfg.tick_size)
                    notional = position["qty"] * exit_price
                    fee = fee_amount(notional, cfg.fee_bp)
                    pnl = position["qty"] * (exit_price - position["entry_price"]) - position["entry_fee"] - fee
                    capital += pnl
                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "entry_time": df.loc[position["entry_idx"], "close_time"],
                        "exit_time": row["close_time"],
                        "side": position["side"],
                        "qty": position["qty"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                    })
                    position = None
                elif position["side"] == "short" and sig > 0:
                    exit_price = apply_slippage(price, side="buy", slippage_ticks=cfg.slippage_ticks, tick_size=cfg.tick_size)
                    notional = position["qty"] * exit_price
                    fee = fee_amount(notional, cfg.fee_bp)
                    pnl = position["qty"] * (position["entry_price"] - exit_price) - position["entry_fee"] - fee
                    capital += pnl
                    trades.append({
                        "entry_idx": position["entry_idx"],
                        "exit_idx": i,
                        "entry_time": df.loc[position["entry_idx"], "close_time"],
                        "exit_time": row["close_time"],
                        "side": position["side"],
                        "qty": position["qty"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                    })
                    position = None
                if position is None:
                    continue

        # Entry logic
        gate_trend = bool(row.get("trend_ok", 1))
        gate_atr = bool(row.get("atr_ok", 1))
        gate_vol = bool(row.get("vol_ok", 1))
        allow = gate_trend and gate_atr and gate_vol

        sig = int(row.get("signal", 0))
        if position is None and allow and sig == 1 and cfg.side in ("long", "both"):
            # Enter long at close
            qty = fixed_fraction(capital=capital, fraction=0.02, price=price)
            entry_price = apply_slippage(price, side="buy", slippage_ticks=cfg.slippage_ticks, tick_size=cfg.tick_size)
            notional = qty * entry_price
            entry_fee = fee_amount(notional, cfg.fee_bp)
            stop = None
            if cfg.stop_mult and cfg.stop_mult > 0:
                stop = initial_stop_atr(entry_price, atr_val, cfg.stop_mult, side="long")
            position = {
                "side": "long",
                "qty": qty,
                "entry_price": entry_price,
                "entry_fee": entry_fee,
                "stop": stop,
                "entry_idx": i,
            }
            continue
        if position is None and allow and sig == -1 and cfg.side in ("short", "both"):
            # Enter short at close
            qty = fixed_fraction(capital=capital, fraction=0.02, price=price)
            entry_price = apply_slippage(price, side="sell", slippage_ticks=cfg.slippage_ticks, tick_size=cfg.tick_size)
            notional = qty * entry_price
            entry_fee = fee_amount(notional, cfg.fee_bp)
            stop = None
            if cfg.stop_mult and cfg.stop_mult > 0:
                stop = initial_stop_atr(entry_price, atr_val, cfg.stop_mult, side="short")
            position = {
                "side": "short",
                "qty": qty,
                "entry_price": entry_price,
                "entry_fee": entry_fee,
                "stop": stop,
                "entry_idx": i,
            }
            continue

    # If open position at the end, close at last close
    if position is not None:
        last = df.iloc[-1]
        if position["side"] == "long":
            exit_price = apply_slippage(float(last["close"]), side="sell", slippage_ticks=cfg.slippage_ticks, tick_size=cfg.tick_size)
            notional = position["qty"] * exit_price
            fee = fee_amount(notional, cfg.fee_bp)
            pnl = position["qty"] * (exit_price - position["entry_price"]) - position["entry_fee"] - fee
        else:
            exit_price = apply_slippage(float(last["close"]), side="buy", slippage_ticks=cfg.slippage_ticks, tick_size=cfg.tick_size)
            notional = position["qty"] * exit_price
            fee = fee_amount(notional, cfg.fee_bp)
            pnl = position["qty"] * (position["entry_price"] - exit_price) - position["entry_fee"] - fee
        capital += pnl
        trades.append(
            {
                "entry_idx": position["entry_idx"],
                "exit_idx": len(df) - 1,
                "entry_time": df.loc[position["entry_idx"], "close_time"],
                "exit_time": last["close_time"],
                "side": position["side"],
                "qty": position["qty"],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
            }
        )

    total_pnl = capital - cfg.capital
    return trades, total_pnl, df
