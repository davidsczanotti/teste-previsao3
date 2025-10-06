from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd

from ...binance_client import get_historical_klines
from .config import MaCrossoverBacktestConfig
from .policy import MASnapshot, ma_signal, decide_action_from_signal


@dataclass
class BacktestResult:
    trades: List[Dict]
    total_pnl: float
    stats: Dict[str, float]
def backtest_ma_crossover(
    df: pd.DataFrame,
    lot_size: float = 0.001,
    fee_rate: float = 0.001,
    min_hold_bars: int = 0,
    cooldown_bars: int = 0,
    allow_short: bool = True,
    initial_capital: float = 10_000.0,
    signal_fn: Optional[Callable[[MASnapshot], int]] = None,
    ma_short_window: int = 7,
    ma_mid_window: int = 40,
    ma_long_window: int = 120,
    ma_type: str = "sma",
) -> BacktestResult:
    """Backtest a 7-candle pattern policy.

    Decisions are made on the close of bar i-1 using candles [i-7, i-1].
    Orders are executed at the open of bar i (next bar), to avoid lookahead bias.
    Single position at a time. Exit on opposite signal (after min_hold_bars).
    """
    df = df.sort_values("Date").reset_index(drop=True)
    longest = max(ma_short_window, ma_mid_window, ma_long_window)
    if len(df) < longest + 2:
        return BacktestResult([], 0.0, {"pnl": 0.0, "num_trades": 0})

    close_series = df["close"].astype(float)
    if ma_type.lower() == "ema":
        ma_short = close_series.ewm(span=ma_short_window, adjust=False).mean()
        ma_mid = close_series.ewm(span=ma_mid_window, adjust=False).mean()
        ma_long = close_series.ewm(span=ma_long_window, adjust=False).mean()
    else:
        ma_short = close_series.rolling(window=ma_short_window, min_periods=ma_short_window).mean()
        ma_mid = close_series.rolling(window=ma_mid_window, min_periods=ma_mid_window).mean()
        ma_long = close_series.rolling(window=ma_long_window, min_periods=ma_long_window).mean()
    ma_short_prev = ma_short.shift(1)
    ma_mid_prev = ma_mid.shift(1)
    ma_long_prev = ma_long.shift(1)

    position = 0  # 0 flat, 1 long, -1 short
    entry_price = 0.0
    realized_pnl = 0.0
    trades: List[Dict] = []
    hold_bars = 0
    cooldown_left = 0
    equity_curve = [initial_capital]

    start_idx = max(ma_long_window, 7)

    # iterate ensuring we have 7 candles ending at i-1 and MA context
    for i in range(start_idx, len(df)):
        last_close = float(df["close"].iloc[i - 1])
        ma_s = float(ma_short.iloc[i]) if not np.isnan(ma_short.iloc[i]) else last_close
        ma_m = float(ma_mid.iloc[i]) if not np.isnan(ma_mid.iloc[i]) else last_close
        ma_l = float(ma_long.iloc[i]) if not np.isnan(ma_long.iloc[i]) else last_close
        ma_s_prev = float(ma_short_prev.iloc[i]) if not np.isnan(ma_short_prev.iloc[i]) else ma_s
        ma_m_prev = float(ma_mid_prev.iloc[i]) if not np.isnan(ma_mid_prev.iloc[i]) else ma_m
        ma_l_prev = float(ma_long_prev.iloc[i]) if not np.isnan(ma_long_prev.iloc[i]) else ma_l

        snapshot = MASnapshot(
            ma_short=ma_s,
            ma_mid=ma_m,
            ma_long=ma_l,
            ma_short_prev=ma_s_prev,
            ma_mid_prev=ma_m_prev,
            ma_long_prev=ma_l_prev,
        )
        signal = signal_fn(snapshot) if signal_fn is not None else ma_signal(snapshot)
        action = decide_action_from_signal(signal, position, allow_short=allow_short)

        # execution at bar i open (or close if open is missing)
        price_open_i = float(df["open"].iloc[i]) if not np.isnan(df["open"].iloc[i]) else float(df["close"].iloc[i])
        t = df["Date"].iloc[i]

        # update cooldown/hold
        if cooldown_left > 0 and position == 0:
            cooldown_left -= 1
        if position != 0:
            hold_bars += 1

        if action == 1 and position == 0 and cooldown_left == 0:
            # open long
            position = 1
            entry_price = price_open_i
            hold_bars = 0
            trades.append({"date": t, "action": "BUY", "price": price_open_i})
            realized_pnl -= fee_rate * price_open_i * lot_size
        elif action == 3 and position == 0 and cooldown_left == 0 and allow_short:
            # open short
            position = -1
            entry_price = price_open_i
            hold_bars = 0
            trades.append({"date": t, "action": "SELL", "price": price_open_i})
            realized_pnl -= fee_rate * price_open_i * lot_size
        elif action == 2 and position == 1 and hold_bars >= min_hold_bars:
            # close long
            pnl = (price_open_i - entry_price) * lot_size
            realized_pnl += pnl
            trades.append({"date": t, "action": "SELL", "price": price_open_i, "pnl": pnl})
            realized_pnl -= fee_rate * price_open_i * lot_size
            position = 0
            entry_price = 0.0
            cooldown_left = cooldown_bars
            hold_bars = 0
        elif action == 4 and position == -1 and hold_bars >= min_hold_bars:
            # close short
            pnl = (entry_price - price_open_i) * lot_size
            realized_pnl += pnl
            trades.append({"date": t, "action": "BUY_TO_COVER", "price": price_open_i, "pnl": pnl})
            realized_pnl -= fee_rate * price_open_i * lot_size
            position = 0
            entry_price = 0.0
            cooldown_left = cooldown_bars
            hold_bars = 0

        # mark-to-market at close of bar i for equity curve
        price_close_i = float(df["close"].iloc[i])
        if position == 1:
            unreal = (price_close_i - entry_price) * lot_size
        elif position == -1:
            unreal = (entry_price - price_close_i) * lot_size
        else:
            unreal = 0.0
        equity_curve.append(initial_capital + realized_pnl + unreal)

    # force close at end
    if position != 0:
        last_price = float(df["close"].iloc[-1])
        t = df["Date"].iloc[-1]
        if position == 1:
            pnl = (last_price - entry_price) * lot_size
            realized_pnl += pnl
            trades.append({"date": t, "action": "SELL (final)", "price": last_price, "pnl": pnl})
            realized_pnl -= fee_rate * last_price * lot_size
        else:
            pnl = (entry_price - last_price) * lot_size
            realized_pnl += pnl
            trades.append({"date": t, "action": "BUY_TO_COVER (final)", "price": last_price, "pnl": pnl})
            realized_pnl -= fee_rate * last_price * lot_size
        equity_curve[-1] = initial_capital + realized_pnl

    closed = [t for t in trades if "pnl" in t]
    n_trades = len(closed)
    wins = len([t for t in closed if t["pnl"] > 0])
    win_rate = (wins / n_trades * 100.0) if n_trades else 0.0
    total_pnl = float(realized_pnl)
    ret_pct = (total_pnl / initial_capital * 100.0) if initial_capital else 0.0

    running_max = np.maximum.accumulate(np.array(equity_curve, dtype=float))
    dd = (np.array(equity_curve) - running_max) / running_max * 100.0
    max_dd_pct = float(dd.min()) if len(dd) else 0.0

    gross_profit = sum(max(t.get("pnl", 0.0), 0.0) for t in closed)
    gross_loss = sum(max(-t.get("pnl", 0.0), 0.0) for t in closed)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_pnl = (total_pnl / n_trades) if n_trades else 0.0

    stats = {
        "pnl": total_pnl,
        "num_trades": n_trades,
        "win_rate": win_rate,
        "return_pct": ret_pct,
        "profit_factor": profit_factor,
        "avg_pnl_per_trade": avg_pnl,
        "max_drawdown_pct": max_dd_pct,
        "fee_rate": fee_rate,
    }

    # Prints resumidos
    print("Backtest MA Crossover:")
    print("Total P&L: $ {0:.2f} ({1:.2f}%)".format(total_pnl, ret_pct))
    print("Trades fechados: {0} | Win rate: {1:.2f}%".format(n_trades, win_rate))
    print("PF: {0:.2f} | Avg/Trade: ${1:.4f} | MDD: {2:.2f}% | Fees: {3:.4%}".format(
        profit_factor if np.isfinite(profit_factor) else float('inf'), avg_pnl, max_dd_pct, fee_rate
    ))

    return BacktestResult(trades, total_pnl, stats)


def _load_df(cfg: MaCrossoverBacktestConfig) -> pd.DataFrame:
    from datetime import datetime, timedelta, UTC
    start_dt = datetime.now(UTC) - timedelta(days=int(cfg.days))
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    df = get_historical_klines(cfg.ticker, cfg.interval, start_str)
    if df.empty:
        raise SystemExit("Falha ao obter dados (cache/Binance) para backtest 7-candle.")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest MA crossover strategy")
    parser.add_argument("--ticker", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--lot_size", type=float, default=0.001)
    parser.add_argument("--fee_rate", type=float, default=0.001)
    parser.add_argument("--min_hold_bars", type=int, default=0)
    parser.add_argument("--cooldown_bars", type=int, default=0)
    parser.add_argument("--ma_short_window", type=int, default=7)
    parser.add_argument("--ma_mid_window", type=int, default=40)
    parser.add_argument("--ma_long_window", type=int, default=120)
    parser.add_argument("--ma_type", choices=["sma", "ema"], default="sma")
    parser.add_argument("--no_short", action="store_true", help="Disable short trades")
    args = parser.parse_args()

    cfg = MaCrossoverBacktestConfig(
        ticker=args.ticker,
        interval=args.interval,
        days=args.days,
        lot_size=args.lot_size,
        fee_rate=args.fee_rate,
        min_hold_bars=args.min_hold_bars,
        cooldown_bars=args.cooldown_bars,
        allow_short=(not args.no_short),
        ma_short_window=args.ma_short_window,
        ma_mid_window=args.ma_mid_window,
        ma_long_window=args.ma_long_window,
        ma_type=args.ma_type,
    )

    df = _load_df(cfg)
    res = backtest_ma_crossover(
        df,
        lot_size=cfg.lot_size,
        fee_rate=cfg.fee_rate,
        min_hold_bars=cfg.min_hold_bars,
        cooldown_bars=cfg.cooldown_bars,
        allow_short=cfg.allow_short,
        ma_short_window=cfg.ma_short_window,
        ma_mid_window=cfg.ma_mid_window,
        ma_long_window=cfg.ma_long_window,
        ma_type=cfg.ma_type,
    )
