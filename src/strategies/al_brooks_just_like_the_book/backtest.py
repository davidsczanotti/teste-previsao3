from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime, timedelta, UTC
from typing import List, Tuple

import numpy as np
import pandas as pd

from ...binance_client import get_historical_klines
from ...utils.data_loader import load_data as _load_data
from .config import AlBrooksBookConfig, load_active_config
from .indicators import add_indicators
from .rules import Signal, detect_signals


def _calc_fills_and_pnl(
    side: str,
    entry_price: float,
    exit_price: float,
    lot_size: float,
    taker_fee_pct: float,
    slippage_pct: float,
) -> tuple[float, float, float, float]:
    """Return (entry_fill, exit_fill, fees, pnl_net)."""
    if side == "long":
        entry_fill = float(entry_price) * (1 + slippage_pct)
        exit_fill = float(exit_price) * (1 - slippage_pct)
        gross = (exit_fill - entry_fill) * lot_size
    else:
        entry_fill = float(entry_price) * (1 - slippage_pct)
        exit_fill = float(exit_price) * (1 + slippage_pct)
        gross = (entry_fill - exit_fill) * lot_size
    fee_entry = entry_fill * lot_size * taker_fee_pct
    fee_exit = exit_fill * lot_size * taker_fee_pct
    fees = fee_entry + fee_exit
    pnl = gross - fees
    return entry_fill, exit_fill, fees, pnl


def _apply_trailing(stop: float, side: str, close_now: float, atr_now: float, trail_mult: float) -> float:
    if not np.isfinite(atr_now) or trail_mult <= 0:
        return stop
    if side == "long":
        trailing = float(close_now) - float(atr_now) * float(trail_mult)
        return max(stop, trailing)
    trailing = float(close_now) + float(atr_now) * float(trail_mult)
    return min(stop, trailing)


def backtest_al_brooks_book(
    df: pd.DataFrame,
    # Context
    ema_fast_period: int = 20,
    ema_medium_period: int = 50,
    ema_slow_period: int = 200,
    slope_lookback: int = 5,
    swing_lookback: int = 3,
    bar_body_min_pct: float = 55.0,
    near_extreme_frac: float = 0.25,
    atr_period: int = 14,
    # Setups
    enable_inside_trend: bool = True,
    enable_h2_l2: bool = True,
    enable_bo_pb: bool = True,
    bo_lookback: int = 20,
    max_ema_distance_atr: float = 1.0,
    use_trend_slope: bool = True,
    min_ema_slope: float = 0.0,
    # Risk mgmt
    risk_reward_ratio: float = 1.4,
    atr_stop_multiplier: float = 0.0,
    atr_trail_multiplier: float = 0.5,
    min_atr: float = 0.0,
    lot_size: float = 0.1,
    # Costs
    taker_fee_pct: float = 0.0004,
    slippage_pct: float = 0.0005,
) -> Tuple[List[dict], float, pd.DataFrame]:
    """Backtests the book-style Al Brooks setups on a OHLCV DataFrame.

    Returns (trades, total_pnl, df_with_indicators).
    """
    if df.empty:
        return [], 0.0, df

    params = {
        "ema_fast_period": ema_fast_period,
        "ema_medium_period": ema_medium_period,
        "ema_slow_period": ema_slow_period,
        "slope_lookback": slope_lookback,
        "swing_lookback": swing_lookback,
        "bar_body_min_pct": bar_body_min_pct,
        "near_extreme_frac": near_extreme_frac,
        "atr_period": atr_period,
        "enable_inside_trend": enable_inside_trend,
        "enable_h2_l2": enable_h2_l2,
        "enable_bo_pb": enable_bo_pb,
        "bo_lookback": bo_lookback,
        "max_ema_distance_atr": max_ema_distance_atr,
        "use_trend_slope": use_trend_slope,
        "min_ema_slope": min_ema_slope,
    }

    df = add_indicators(df, params)

    trades: List[dict] = []
    position = None  # "long" or "short"

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Manage open trade
        if position:
            trade = trades[-1]
            # Trailing stop using ATR on current bar close
            if atr_trail_multiplier > 0 and np.isfinite(row.get("atr", np.nan)):
                trade["stop_loss"] = _apply_trailing(
                    trade["stop_loss"], position, row["close"], row["atr"], atr_trail_multiplier
                )

            # Check exits using current bar extremes
            exit_price = None
            exit_reason = None
            if position == "long":
                if row["low"] <= trade["stop_loss"]:
                    exit_price = trade["stop_loss"]
                    exit_reason = "stop"
                elif row["high"] >= trade["take_profit"]:
                    exit_price = trade["take_profit"]
                    exit_reason = "target"
            else:
                if row["high"] >= trade["stop_loss"]:
                    exit_price = trade["stop_loss"]
                    exit_reason = "stop"
                elif row["low"] <= trade["take_profit"]:
                    exit_price = trade["take_profit"]
                    exit_reason = "target"

            if exit_price is not None:
                entry_price = float(trade["entry_price"])
                entry_fill, exit_fill, fees, pnl = _calc_fills_and_pnl(
                    position, entry_price, float(exit_price), lot_size, taker_fee_pct, slippage_pct
                )
                trade.update(
                    {
                        "exit_price": float(exit_price),
                        "exit_date": row["Date"],
                        "entry_fill": entry_fill,
                        "exit_fill": exit_fill,
                        "fees": fees,
                        "pnl": pnl,
                        "exit_reason": exit_reason,
                    }
                )
                position = None
                continue

        # If still in trade, skip new signals
        if position is not None:
            continue

        # Volatility/ATR guard
        if not np.isfinite(prev.get("atr", np.nan)) or float(prev["atr"]) <= float(min_atr):
            continue

        # Detect signals on last closed bar (i-1)
        sigs = detect_signals(df, i - 1, params)
        if not sigs:
            continue

        # Choose first signal by priority (already ordered in detect_signals)
        s: Signal = sigs[0]

        # Confirm stop entry with the current bar extremes
        triggered = False
        if s.direction == "long" and row["high"] >= s.entry:
            triggered = True
        elif s.direction == "short" and row["low"] <= s.entry:
            triggered = True

        if not triggered:
            continue

        atr_val = float(prev.get("atr", np.nan))
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue

        # Compute stop/target
        if float(atr_stop_multiplier) > 0:
            if s.direction == "long":
                stop_loss = float(s.entry) - atr_val * float(atr_stop_multiplier)
            else:
                stop_loss = float(s.entry) + atr_val * float(atr_stop_multiplier)
        else:
            stop_loss = float(s.stop)

        if s.direction == "long":
            risk = float(s.entry) - float(stop_loss)
            if risk <= 0:
                continue
            take_profit = float(s.entry) + risk * float(risk_reward_ratio)
        else:
            risk = float(stop_loss) - float(s.entry)
            if risk <= 0:
                continue
            take_profit = float(s.entry) - risk * float(risk_reward_ratio)

        trades.append(
            {
                "entry_date": row["Date"],
                "entry_price": float(s.entry),
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "type": s.direction,
                "signal": s.name,
                "reason": s.reason,
                "initial_risk": float(risk),
                "atr": float(atr_val),
            }
        )
        position = s.direction

    # Close open position at the last close
    if trades and position:
        trade = trades[-1]
        final_price = float(df["close"].iloc[-1])
        entry_price = float(trade["entry_price"])
        entry_fill, exit_fill, fees, pnl = _calc_fills_and_pnl(
            position, entry_price, final_price, lot_size, taker_fee_pct, slippage_pct
        )
        trade.update(
            {
                "exit_price": final_price,
                "exit_date": df["Date"].iloc[-1],
                "entry_fill": entry_fill,
                "exit_fill": exit_fill,
                "fees": fees,
                "pnl": pnl,
                "exit_reason": "eod",
            }
        )
        position = None

    for t in trades:
        t.setdefault("pnl", 0.0)

    total_pnl = float(sum(t["pnl"] for t in trades))
    return trades, total_pnl, df


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest: Al Brooks Book-Style strategy")
    ap.add_argument("--ticker", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--lot-size", type=float, default=0.1)
    ap.add_argument("--cache-only", action="store_true", help="Usa apenas o cache local, sem chamadas à rede")
    args = ap.parse_args()

    cfg = load_active_config(args.ticker, args.interval)
    if cfg:
        print(f"Using ACTIVE config for {args.ticker}@{args.interval}")
        params = asdict(cfg)
        for k in ("ticker", "interval", "days"):
            params.pop(k, None)
        load_days = cfg.days
        lot_size = cfg.lot_size
    else:
        print("No ACTIVE config found. Using defaults.")
        params = {}
        load_days = args.days
        lot_size = args.lot_size

    print(f"Loading data: {args.ticker}@{args.interval} for {load_days} days...")
    if args.cache_only:
        try:
            df = _load_data(args.ticker, args.interval, days=load_days, use_cache_only=True)
        except Exception as e:
            print(f"Failed to load from cache-only: {e}")
            df = pd.DataFrame()
    else:
        start_dt = datetime.now(UTC) - timedelta(days=load_days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        df = get_historical_klines(args.ticker, args.interval, start_str)
    if df.empty:
        print("No data returned.")
        return
    print(f"Candles loaded: {len(df)}")

    print("Running backtest...")
    trades, pnl, _ = backtest_al_brooks_book(df.copy(), lot_size=lot_size, **params)

    closed = [t for t in trades if "pnl" in t]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    wr = (len(wins) / len(closed) * 100.0) if closed else 0.0
    total_profit = sum(t["pnl"] for t in wins)
    total_loss = -sum(t["pnl"] for t in losses)
    pf = (total_profit / total_loss) if total_loss > 0 else float("inf")

    print("\n--- Backtest Results ---")
    print(f"Final P&L: ${pnl:.2f} | Trades: {len(closed)} | WinRate: {wr:.2f}% | PF: {pf:.2f}")

    # Optionally write trades CSV for audit
    out_dir = "reports/live"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ALBROOKS_BOOK_{args.ticker}_{args.interval}_trades.csv")
    pd.DataFrame(closed).to_csv(out_path, index=False)
    print(f"Trades saved to: {out_path}")


if __name__ == "__main__":
    main()
