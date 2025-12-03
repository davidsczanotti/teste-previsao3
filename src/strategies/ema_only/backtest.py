from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class EmaOnlyParams:
    ema_period: int = 8
    slow_ema_period: int | None = None  # required for ema_cross mode
    trend_filter_period: int | None = None
    use_trend_filter: bool = False
    pullback_pct: float = 0.0  # extra distance below EMA required to consider entry
    ref_filter_enabled: bool = False  # use higher timeframe EMA as bias
    ref_ema_period: int | None = None  # used for reporting only; values come merged in df
    ref_buffer_pct: float = 0.0  # tolerance above ref EMA
    ref_timeframe: str | None = None  # reporting
    lot_size: float = 0.001  # BTC quantity for trades
    fee_rate: float = 0.001  # 0.1% per side
    use_cross: bool = False  # price/EMA reclaim entry in price_reversion mode
    signal_mode: str = "price_reversion"  # price_reversion | ema_cross


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
    ema_fast = compute_ema(closes, params.ema_period)
    ema_slow = compute_ema(closes, params.slow_ema_period) if params.slow_ema_period else None
    ema_trend = compute_ema(closes, params.trend_filter_period) if params.trend_filter_period else None
    ref_ema_col = df["ref_ema"].astype(float) if params.ref_filter_enabled and "ref_ema" in df.columns else None

    if params.signal_mode not in {"price_reversion", "ema_cross"}:
        raise ValueError(f"signal_mode desconhecido: {params.signal_mode}")

    # Start after EMA warms up
    start = max(params.ema_period + 1, params.slow_ema_period + 1 if params.slow_ema_period else 0, params.trend_filter_period + 1 if params.trend_filter_period else 0, 2)

    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    realized_pnl = 0.0
    equity_curve = [initial_capital]
    trades: List[Dict] = []

    for i in range(start, len(df)):
        p_prev = float(closes.iloc[i - 1])
        e_prev = float(ema_fast.iloc[i - 1]) if not np.isnan(ema_fast.iloc[i - 1]) else np.nan
        p = float(closes.iloc[i])
        e = float(ema_fast.iloc[i]) if not np.isnan(ema_fast.iloc[i]) else np.nan
        t = df["Date"].iloc[i]
        s_prev = float(ema_slow.iloc[i - 1]) if ema_slow is not None and not np.isnan(ema_slow.iloc[i - 1]) else np.nan
        s = float(ema_slow.iloc[i]) if ema_slow is not None and not np.isnan(ema_slow.iloc[i]) else np.nan
        tr_prev = (
            float(ema_trend.iloc[i - 1])
            if ema_trend is not None and not np.isnan(ema_trend.iloc[i - 1])
            else np.nan
        )
        tr_now = float(ema_trend.iloc[i]) if ema_trend is not None and not np.isnan(ema_trend.iloc[i]) else np.nan
        ref_prev = float(ref_ema_col.iloc[i - 1]) if ref_ema_col is not None and not np.isnan(ref_ema_col.iloc[i - 1]) else np.nan
        ref_now = float(ref_ema_col.iloc[i]) if ref_ema_col is not None and not np.isnan(ref_ema_col.iloc[i]) else np.nan

        if np.isnan(e) or np.isnan(e_prev):
            equity_curve.append(initial_capital + realized_pnl)
            continue

        ref_ok = True
        if params.ref_filter_enabled:
            if np.isnan(ref_prev) or np.isnan(ref_now):
                equity_curve.append(initial_capital + realized_pnl)
                continue
            buffer = 1.0 + params.ref_buffer_pct
            ref_ok = (p > ref_now * buffer) and (e > ref_now * buffer)

        if params.signal_mode == "ema_cross":
            if ema_slow is None:
                raise ValueError("signal_mode='ema_cross' requer slow_ema_period definido.")
            cross_up = (e_prev <= s_prev) and (e > s)
            cross_down = (e_prev >= s_prev) and (e < s)

            if position == 0 and cross_up and ref_ok:
                position = 1
                entry_price = p
                trades.append({"date": t, "action": "BUY", "price": p})
                realized_pnl -= params.fee_rate * p * params.lot_size
            elif position == 1 and (cross_down or (params.ref_filter_enabled and not ref_ok)):
                pnl = (p - entry_price) * params.lot_size
                realized_pnl += pnl
                trades.append({"date": t, "action": "SELL", "price": p, "pnl": pnl})
                realized_pnl -= params.fee_rate * p * params.lot_size
                position = 0
                entry_price = 0.0

        else:  # price_reversion default
            trend_ok = True
            if params.use_trend_filter:
                if ema_trend is None:
                    raise ValueError("use_trend_filter=True requer trend_filter_period configurado.")
                if np.isnan(tr_prev) or np.isnan(tr_now):
                    equity_curve.append(initial_capital + realized_pnl)
                    continue
                trend_ok = (e_prev > tr_prev) and (p_prev > tr_prev)

            pullback_level_prev = e_prev * (1 - params.pullback_pct)

            if position == 0:
                if params.use_cross:
                    enter = trend_ok and ref_ok and (p_prev <= pullback_level_prev) and (p > e)
                else:
                    enter = trend_ok and ref_ok and (p < pullback_level_prev)
                if enter:
                    position = 1
                    entry_price = p
                    trades.append({"date": t, "action": "BUY", "price": p})
                    realized_pnl -= params.fee_rate * p * params.lot_size

            else:  # position == 1
                exit_price = (p < e) if params.use_cross else (p > e)
                exit_trend = params.use_trend_filter and ema_trend is not None and (e < tr_now)
                exit_ref = params.ref_filter_enabled and not ref_ok
                if exit_price or exit_trend:
                    pnl = (p - entry_price) * params.lot_size
                    realized_pnl += pnl
                    trades.append({"date": t, "action": "SELL", "price": p, "pnl": pnl})
                    realized_pnl -= params.fee_rate * p * params.lot_size
                    position = 0
                    entry_price = 0.0
                elif exit_ref:
                    pnl = (p - entry_price) * params.lot_size
                    realized_pnl += pnl
                    trades.append({"date": t, "action": "SELL", "price": p, "pnl": pnl, "reason": "ref_filter"})
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
    equity_arr = np.array(equity_curve, dtype=float)
    rets = np.diff(equity_arr) / equity_arr[:-1] if len(equity_arr) > 1 else np.array([])
    total_seconds = (df["Date"].iloc[-1] - df["Date"].iloc[0]).total_seconds() if len(df) > 1 else 0.0
    seconds_per_bar = total_seconds / max(1, len(equity_arr) - 1)
    bars_per_year = (365 * 24 * 3600) / seconds_per_bar if seconds_per_bar > 0 else 0.0
    ann_factor = np.sqrt(bars_per_year) if bars_per_year > 0 else 0.0
    sharpe = (rets.mean() / rets.std() * ann_factor) if len(rets) > 1 and rets.std() > 0 else 0.0
    neg_rets = rets[rets < 0]
    downside_std = neg_rets.std() if len(neg_rets) > 0 else 0.0
    sortino = (rets.mean() / downside_std * ann_factor) if downside_std > 0 else 0.0
    total_ret_dec = equity_arr[-1] / equity_arr[0] - 1 if len(equity_arr) > 1 else 0.0
    years = total_seconds / (365 * 24 * 3600) if total_seconds > 0 else 0.0
    annual_ret = total_ret_dec / years if years > 0 else 0.0
    max_dd_dec = max_dd_pct / 100.0
    calmar = (annual_ret / abs(max_dd_dec)) if max_dd_dec != 0 else 0.0

    stats = {
        "pnl": total_pnl,
        "num_trades": n_trades,
        "win_rate": win_rate,
        "return_pct": ret_pct,
        "avg_pnl_per_trade": avg_pnl,
        "max_drawdown_pct": max_dd_pct,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "fee_rate": params.fee_rate,
        "ema_period": params.ema_period,
        "use_cross": params.use_cross,
        "lot_size": params.lot_size,
        "slow_ema_period": params.slow_ema_period,
        "trend_filter_period": params.trend_filter_period,
        "use_trend_filter": params.use_trend_filter,
        "pullback_pct": params.pullback_pct,
        "signal_mode": params.signal_mode,
        "ref_filter_enabled": params.ref_filter_enabled,
        "ref_ema_period": params.ref_ema_period,
        "ref_buffer_pct": params.ref_buffer_pct,
        "ref_timeframe": params.ref_timeframe,
    }

    return trades, total_pnl, stats
