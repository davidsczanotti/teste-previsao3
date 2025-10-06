from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from ...binance_client import get_historical_klines
from .config import load_active_config


@dataclass
class DonchianParams:
    window_high: int = 20
    window_low: int = 20
    atr_period: int = 14
    atr_mult: float = 2.0
    use_ema: bool = True
    ema_period: int = 200
    allow_short: bool = False


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def backtest_donchian(
    df: pd.DataFrame,
    params: DonchianParams = DonchianParams(),
    initial_capital: float = 10_000.0,
    lot_size: float = 0.1,
    fee_rate: float = 0.001,
) -> Tuple[List[Dict], float, Dict[str, float]]:
    df = df.sort_values("Date").reset_index(drop=True)
    closes = df["close"].astype(float)
    ema = None
    if params.use_ema and params.ema_period > 0:
        ema = closes.ewm(span=params.ema_period, adjust=False).mean()
    a = atr(df, params.atr_period)

    hh = df["high"].rolling(window=params.window_high, min_periods=params.window_high).max().shift(1)
    ll = df["low"].rolling(window=params.window_low, min_periods=params.window_low).min().shift(1)

    start = max(params.window_high, params.window_low, params.atr_period, (params.ema_period if params.use_ema else 0)) + 2

    position = 0  # 1 long, -1 short, 0 none
    entry_price = 0.0
    trail = np.nan
    realized_pnl = 0.0
    equity_curve = [initial_capital]
    trades: List[Dict] = []
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(start, len(df)):
        price = float(closes.iloc[i])
        t = df["Date"].iloc[i]
        hh_i = float(hh.iloc[i]) if not np.isnan(hh.iloc[i]) else None
        ll_i = float(ll.iloc[i]) if not np.isnan(ll.iloc[i]) else None
        atr_i = float(a.iloc[i]) if not np.isnan(a.iloc[i]) else None
        ema_ok_long = True
        ema_ok_short = True
        if ema is not None and not np.isnan(ema.iloc[i]):
            ema_val = float(ema.iloc[i])
            ema_ok_long = price > ema_val
            ema_ok_short = price < ema_val

        # Signals
        long_break = (hh_i is not None) and (price > hh_i) and ema_ok_long
        short_break = (ll_i is not None) and (price < ll_i) and ema_ok_short and params.allow_short

        if position == 0:
            if long_break and atr_i is not None:
                position = 1
                entry_price = price
                trail = price - params.atr_mult * atr_i
                trades.append({"date": t, "action": "BUY", "price": price})
                realized_pnl -= fee_rate * price * lot_size
            elif short_break and atr_i is not None:
                position = -1
                entry_price = price
                trail = price + params.atr_mult * atr_i
                trades.append({"date": t, "action": "SELL", "price": price})
                realized_pnl -= fee_rate * price * lot_size

        elif position == 1:
            # Update trailing stop upwards
            if atr_i is not None:
                new_trail = price - params.atr_mult * atr_i
                trail = max(trail, new_trail) if not np.isnan(trail) else new_trail
            stop_hit = price <= trail if not np.isnan(trail) else False
            if stop_hit:
                pnl = (price - entry_price) * lot_size
                realized_pnl += pnl
                trades.append({"date": t, "action": "SELL", "price": price, "pnl": pnl})
                realized_pnl -= fee_rate * price * lot_size
                if pnl >= 0:
                    gross_profit += pnl
                else:
                    gross_loss += -pnl
                position = 0
                entry_price = 0.0
                trail = np.nan

        elif position == -1:
            if atr_i is not None:
                new_trail = price + params.atr_mult * atr_i
                trail = min(trail, new_trail) if not np.isnan(trail) else new_trail
            stop_hit = price >= trail if not np.isnan(trail) else False
            if stop_hit:
                pnl = (entry_price - price) * lot_size
                realized_pnl += pnl
                trades.append({"date": t, "action": "BUY_TO_COVER", "price": price, "pnl": pnl})
                realized_pnl -= fee_rate * price * lot_size
                if pnl >= 0:
                    gross_profit += pnl
                else:
                    gross_loss += -pnl
                position = 0
                entry_price = 0.0
                trail = np.nan

        # Equity
        if position == 1:
            unreal = (price - entry_price) * lot_size
        elif position == -1:
            unreal = (entry_price - price) * lot_size
        else:
            unreal = 0.0
        equity_curve.append(initial_capital + realized_pnl + unreal)

    # Close any open at the end
    if position != 0:
        last_price = float(closes.iloc[-1])
        t = df["Date"].iloc[-1]
        if position == 1:
            pnl = (last_price - entry_price) * lot_size
            realized_pnl += pnl
            trades.append({"date": t, "action": "SELL (final)", "price": last_price, "pnl": pnl})
            realized_pnl -= fee_rate * last_price * lot_size
            if pnl >= 0:
                gross_profit += pnl
            else:
                gross_loss += -pnl
        else:
            pnl = (entry_price - last_price) * lot_size
            realized_pnl += pnl
            trades.append({"date": t, "action": "BUY_TO_COVER (final)", "price": last_price, "pnl": pnl})
            realized_pnl -= fee_rate * last_price * lot_size
            if pnl >= 0:
                gross_profit += pnl
            else:
                gross_loss += -pnl
        equity_curve[-1] = initial_capital + realized_pnl

    # Metrics
    total_pnl = float(realized_pnl)
    closed = [t for t in trades if "pnl" in t]
    n_trades = len(closed)
    wins = len([t for t in closed if t["pnl"] > 0])
    win_rate = (wins / n_trades * 100.0) if n_trades else 0.0
    ret_pct = (total_pnl / initial_capital * 100.0) if initial_capital else 0.0
    running_max = np.maximum.accumulate(np.array(equity_curve, dtype=float))
    dd = (np.array(equity_curve) - running_max) / running_max * 100.0
    max_dd_pct = float(dd.min()) if len(dd) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    avg_pnl = (total_pnl / n_trades) if n_trades else 0.0

    stats = {
        "pnl": total_pnl,
        "num_trades": n_trades,
        "win_rate": win_rate,
        "return_pct": ret_pct,
        "profit_factor": profit_factor,
        "avg_pnl_per_trade": avg_pnl,
        "max_drawdown_pct": max_dd_pct,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "fee_rate": fee_rate,
    }

    print(
        "Donchian Backtest (H={h}, L={l}, ATRp={ap}, ATRx={am}, EMA{use}/{ep}, short={sh}):".format(
            h=params.window_high,
            l=params.window_low,
            ap=params.atr_period,
            am=params.atr_mult,
            use="on" if params.use_ema else "off",
            ep=params.ema_period,
            sh=params.allow_short,
        )
    )
    print("Total P&L: $ {0:.2f} ({1:.2f}%)".format(total_pnl, ret_pct))
    print("Trades: {0} | Win rate: {1:.2f}% | PF: {2:.2f} | MDD: {3:.2f}%".format(n_trades, win_rate, profit_factor if np.isfinite(profit_factor) else float('inf'), max_dd_pct))

    return trades, total_pnl, stats


if __name__ == "__main__":
    from datetime import datetime, timedelta, UTC

    default_ticker = "BTCUSDT"
    default_interval = "15m"
    cfg = load_active_config(default_ticker, default_interval)
    if cfg:
        days = cfg.days if cfg.days else 90
        start_dt = datetime.now(UTC) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Usando config ativa DONCH para {cfg.ticker}@{cfg.interval}. Baixando {days} dias...")
        df = get_historical_klines(cfg.ticker, cfg.interval, start_str)
        p = DonchianParams(
            window_high=cfg.window_high,
            window_low=cfg.window_low,
            atr_period=cfg.atr_period,
            atr_mult=cfg.atr_mult,
            use_ema=cfg.use_ema,
            ema_period=cfg.ema_period,
            allow_short=cfg.allow_short,
        )
        backtest_donchian(df, params=p, initial_capital=1_000.0, lot_size=cfg.lot_size or 0.001, fee_rate=cfg.fee_rate)
    else:
        # Default quick run
        from datetime import datetime, timedelta, UTC
        start_dt = datetime.now(UTC) - timedelta(days=90)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        df = get_historical_klines(default_ticker, default_interval, start_str)
        p = DonchianParams(window_high=20, window_low=20, atr_period=14, atr_mult=2.0, use_ema=True, ema_period=200)
        backtest_donchian(df, params=p, initial_capital=1_000.0, lot_size=0.001, fee_rate=0.0005)

