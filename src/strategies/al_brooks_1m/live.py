from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path
import csv

import numpy as np
import pandas as pd

# Safe, headless backend for PNG generation
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...binance_client import get_current_price, get_historical_klines
from .config import load_active_config
from .indicators import add_indicators

position_state = {
    "position": None,
    "entry_price": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "capital": 100.0,
    "entry_fill": 0.0,  # preço de execução com slippage
}

PULLBACK_LOOKBACK = 10

# In-memory trade events for plotting entries/exits during this session
TRADE_EVENTS: list[dict] = []
TRADES_CSV_PATH: Path | None = None


def compute_signal(df: pd.DataFrame, params: dict) -> tuple[str, str]:
    """Determina se há sinal de compra ou venda no último candle fechado."""
    if len(df) < 3:
        return "hold", "Dados insuficientes"

    last = df.iloc[-2]  # último candle fechado

    if last["avg_deviation_pct"] > params["max_avg_deviation_pct"]:
        return "hold", f"Preço esticado ({last['avg_deviation_pct']:.2f}%)"

    if np.isnan(last["atr"]) or last["atr"] <= params.get("min_atr", 0.0):
        return "hold", f"ATR baixo ({last['atr']:.2f})"

    if np.isnan(last["adx"]) or last["adx"] < params["adx_threshold"]:
        return "hold", f"ADX baixo ({last['adx']:.1f})"

    allow_long = True
    allow_short = True
    if params.get("use_htf_bias", True):
        bias = last.get("trend_bias")
        if not np.isnan(bias):
            allow_long = bias >= 0
            allow_short = bias <= 0

    uptrend = (
        last["close"] > last["ema_medium"]
        and last["ema_fast"] > last["ema_medium"]
        and last["ema_medium"] > last["ema_slow"]
    )
    downtrend = (
        last["close"] < last["ema_medium"]
        and last["ema_fast"] < last["ema_medium"]
        and last["ema_medium"] < last["ema_slow"]
    )

    if not last["is_inside_bar"]:
        return "hold", "Não é Inside Bar"

    if allow_long and uptrend and last["close"] < last["ema_fast"]:
        return "buy", "Sinal de compra"

    if allow_short and downtrend and last["close"] > last["ema_fast"]:
        return "sell", "Sinal de venda"

    return "hold", "Sem alinhamento de EMAs/pullback"


def calculate_levels(
    df: pd.DataFrame, params: dict, direction: str, entry_price: float
) -> tuple[float, float] | tuple[None, None]:
    """Calcula stop loss e take profit baseados em ATR e price action recente."""
    last_closed = df.iloc[-2]
    atr_value = last_closed["atr"]
    if np.isnan(atr_value) or atr_value <= 0:
        return None, None

    lookback_slice = df.iloc[-(PULLBACK_LOOKBACK + 1) : -1]

    if direction == "long":
        pullback_low = lookback_slice["low"].min()
        stop_candidates = [
            entry_price - atr_value * params["atr_stop_multiplier"],
            pullback_low,
        ]
        stop_loss = min(stop_candidates)
        risk = entry_price - stop_loss
        if risk <= 0:
            return None, None
        take_profit = entry_price + risk * params["risk_reward_ratio"]
    else:
        rally_high = lookback_slice["high"].max()
        stop_candidates = [
            entry_price + atr_value * params["atr_stop_multiplier"],
            rally_high,
        ]
        stop_loss = max(stop_candidates)
        risk = stop_loss - entry_price
        if risk <= 0:
            return None, None
        take_profit = entry_price - risk * params["risk_reward_ratio"]

    return stop_loss, take_profit


def handle_exit(exit_type: str, price: float, params: dict, event_time) -> None:
    """Fecha a posição atual e atualiza capital."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    entry_price = position_state["entry_price"]
    lot_size = params["lot_size"]
    fee_pct = float(params.get("taker_fee_pct", 0.0))
    slip_pct = float(params.get("slippage_pct", 0.0))

    side = position_state.get("position")
    # Calcula fills com slippage e P&L líquido de taxas
    if side == "long":
        entry_fill = float(position_state.get("entry_fill") or (entry_price * (1 + slip_pct)))
        exit_fill = float(price) * (1 - slip_pct)
        gross = (exit_fill - entry_fill) * lot_size
    else:
        entry_fill = float(position_state.get("entry_fill") or (entry_price * (1 - slip_pct)))
        exit_fill = float(price) * (1 + slip_pct)
        gross = (entry_fill - exit_fill) * lot_size
    fee_entry = entry_fill * lot_size * fee_pct
    fee_exit = exit_fill * lot_size * fee_pct
    pnl = gross - fee_entry - fee_exit

    position_state["capital"] += pnl
    print(
        f"[{now_str}] PREÇO: {price:.2f} | SAÍDA: {exit_type} | P&L (net): ${pnl:.2f} | CAPITAL: ${position_state['capital']:.2f}"
    )
    # Log exit marker for plotting
    try:
        t = pd.to_datetime(event_time)
        # Ensure naive datetime to align with DF axis
        if getattr(t, "tzinfo", None) is not None:
            t = t.tz_localize(None)
    except Exception:
        t = datetime.now().replace(tzinfo=None)
    # Armazena o preço de execução (fill) e o PnL para uso no gráfico
    TRADE_EVENTS.append({
        "type": "exit",
        "side": side or "",
        "time": t,
        "price": float(exit_fill),
        "label": exit_type,
        "pnl": float(pnl),
        "capital": float(position_state.get("capital", 0.0)),
    })
    # Persist trade to CSV
    # Persiste evento de saída usando o preço de execução
    _append_trade(
        event_time=t,
        event_type="exit",
        subtype=exit_type,
        side=side or "",
        price=float(exit_fill),
        entry_price=float(entry_price),
        stop_loss=float(position_state.get("stop_loss", 0.0)),
        take_profit=float(position_state.get("take_profit", 0.0)),
        capital=float(position_state.get("capital", 0.0)),
        reason="",
    )
    position_state.update({
        "position": None,
        "entry_price": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "entry_fill": 0.0,
    })


def manage_existing_position(df: pd.DataFrame, current_price: float, params: dict) -> None:
    """Atualiza posição aberta, aplicando trailing stop e verificando saídas."""
    position = position_state["position"]
    if not position:
        return

    last = df.iloc[-1]
    atr_value = last.get("atr", np.nan)
    trail_mult = params.get("atr_trail_multiplier", 0.0)

    if trail_mult > 0 and not np.isnan(atr_value):
        # Alinhar com o backtest: trailing baseado no close do candle atual
        close_now = last.get("close", np.nan)
        if not np.isnan(close_now):
            if position == "long":
                trailing = close_now - atr_value * trail_mult
                position_state["stop_loss"] = max(position_state["stop_loss"], trailing)
            else:
                trailing = close_now + atr_value * trail_mult
                position_state["stop_loss"] = min(position_state["stop_loss"], trailing)

    # Use current candle extremes to avoid missing touches between polls
    cur_candle = df.iloc[-1]
    cur_high = cur_candle.get("high", np.nan)
    cur_low = cur_candle.get("low", np.nan)

    fee_pct = float(params.get("taker_fee_pct", 0.0))
    slip_pct = float(params.get("slippage_pct", 0.0))

    if position == "long":
        if not np.isnan(cur_low) and cur_low <= position_state["stop_loss"]:
            handle_exit("STOP LOSS", position_state["stop_loss"], params, df.iloc[-1]["Date"])
        elif not np.isnan(cur_high) and cur_high >= position_state["take_profit"]:
            handle_exit("TAKE PROFIT", position_state["take_profit"], params, df.iloc[-1]["Date"])
        else:
            entry_fill = float(position_state.get("entry_fill") or (position_state["entry_price"] * (1 + slip_pct)))
            exit_fill_now = float(current_price) * (1 - slip_pct)
            gross = (exit_fill_now - entry_fill) * params["lot_size"]
            fees = (entry_fill + exit_fill_now) * params["lot_size"] * fee_pct
            unrealized = gross - fees
            print(
                f"PREÇO: {current_price:.2f} | POSIÇÃO: LONG | STOP: {position_state['stop_loss']:.2f} | "
                f"ALVO: {position_state['take_profit']:.2f} | P&L flutuante (net): ${unrealized:.2f}"
            )
    else:
        if not np.isnan(cur_high) and cur_high >= position_state["stop_loss"]:
            handle_exit("STOP LOSS", position_state["stop_loss"], params, df.iloc[-1]["Date"])
        elif not np.isnan(cur_low) and cur_low <= position_state["take_profit"]:
            handle_exit("TAKE PROFIT", position_state["take_profit"], params, df.iloc[-1]["Date"])
        else:
            entry_fill = float(position_state.get("entry_fill") or (position_state["entry_price"] * (1 - slip_pct)))
            exit_fill_now = float(current_price) * (1 + slip_pct)
            gross = (entry_fill - exit_fill_now) * params["lot_size"]
            fees = (entry_fill + exit_fill_now) * params["lot_size"] * fee_pct
            unrealized = gross - fees
            print(
                f"PREÇO: {current_price:.2f} | POSIÇÃO: SHORT | STOP: {position_state['stop_loss']:.2f} | "
                f"ALVO: {position_state['take_profit']:.2f} | P&L flutuante (net): ${unrealized:.2f}"
            )


def _ensure_live_paths(ticker: str, interval: str) -> tuple[Path, Path]:
    """Ensures reports/live exists and returns (csv_path, png_path). Overwrites CSV header at start."""
    live_dir = Path("reports") / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ALBROOKS_{ticker}_{interval}"
    csv_path = live_dir / f"{stem}.csv"
    png_path = live_dir / f"{stem}.png"
    return csv_path, png_path


def _ensure_trades_csv(ticker: str, interval: str) -> Path:
    live_dir = Path("reports") / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ALBROOKS_{ticker}_{interval}_trades.csv"
    path = live_dir / stem
    if not path.exists():
        header = [
            "timestamp_utc",
            "event_time",
            "type",
            "subtype",
            "side",
            "price",
            "entry_price",
            "stop_loss",
            "take_profit",
            "capital",
            "reason",
        ]
        path.write_text(",".join(header) + "\n", encoding="utf-8")
    return path


def _init_csv(csv_path: Path) -> None:
    """(Re)creates the CSV file with header for auditing."""
    header = [
        "timestamp_utc",
        "candle_time",
        "price",
        "signal",
        "reason",
        "position",
        "entry_price",
        "stop_loss",
        "take_profit",
        "capital",
        "ema_fast",
        "ema_medium",
        "ema_slow",
        "is_inside_bar",
        "avg_deviation_pct",
        "adx",
        "atr",
        "trend_bias",
        # Auditoria adicional
        "close_last_closed",
        "high_last_closed",
        "low_last_closed",
        "high_current",
        "low_current",
        "uptrend",
        "downtrend",
        "pullback_long_ok",
        "pullback_short_ok",
        "allow_long",
        "allow_short",
        "adx_ok",
        "atr_ok",
        "deviation_ok",
        # Limiares de parâmetros
        "adx_threshold",
        "min_atr",
        "max_avg_deviation_pct",
        # Sinal/rompimento (auditoria rápida)
        "setup_ok",
        "entry_confirmed",
        # Novos campos p/ auditoria offline
        "prev_high_for_entry",
        "prev_low_for_entry",
        "would_enter_backtest",
    ]
    csv_path.write_text(",".join(header) + "\n", encoding="utf-8")


def _append_csv(
    csv_path: Path,
    last_closed: pd.Series,
    current_price: float,
    signal: str,
    reason: str,
    capital: float,
    cur_high: float,
    cur_low: float,
    params: dict,
) -> None:
    # Deriva campos adicionais conforme a lógica de compute_signal
    c = last_closed.get("close")
    h = last_closed.get("high")
    l = last_closed.get("low")
    ema_f = last_closed.get("ema_fast")
    ema_m = last_closed.get("ema_medium")
    ema_s = last_closed.get("ema_slow")

    def _n(x):
        return (x is not None) and (not np.isnan(x))

    uptrend = _n(c) and _n(ema_f) and _n(ema_m) and _n(ema_s) and (c > ema_m) and (ema_f > ema_m) and (ema_m > ema_s)
    downtrend = _n(c) and _n(ema_f) and _n(ema_m) and _n(ema_s) and (c < ema_m) and (ema_f < ema_m) and (ema_m < ema_s)
    pullback_long_ok = _n(c) and _n(ema_f) and (c < ema_f)
    pullback_short_ok = _n(c) and _n(ema_f) and (c > ema_f)

    allow_long = True
    allow_short = True
    if params.get("use_htf_bias", True):
        bias = last_closed.get("trend_bias")
        if _n(bias):
            allow_long = bias >= 0
            allow_short = bias <= 0

    # Filter checks vs parameters
    avg_dev = last_closed.get("avg_deviation_pct")
    adx_val = last_closed.get("adx")
    atr_val = last_closed.get("atr")
    deviation_ok = _n(avg_dev) and (avg_dev <= params["max_avg_deviation_pct"])
    adx_ok = _n(adx_val) and (adx_val >= params["adx_threshold"])
    atr_ok = _n(atr_val) and (atr_val > params.get("min_atr", 0.0))

    # Setup directional checks
    is_inside_bool = bool(last_closed.get("is_inside_bar", False))
    setup_long = (
        signal == "buy"
        and deviation_ok
        and adx_ok
        and atr_ok
        and is_inside_bool
        and allow_long
        and uptrend
        and pullback_long_ok
    )
    setup_short = (
        signal == "sell"
        and deviation_ok
        and adx_ok
        and atr_ok
        and is_inside_bool
        and allow_short
        and downtrend
        and pullback_short_ok
    )
    setup_ok = setup_long or setup_short

    # Breakout confirmation with current candle extremes
    entry_breakout = False
    if signal == "buy" and _n(h) and (cur_high is not None) and (not np.isnan(cur_high)):
        entry_breakout = cur_high >= h
    elif signal == "sell" and _n(l) and (cur_low is not None) and (not np.isnan(cur_low)):
        entry_breakout = cur_low <= l

    row = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "candle_time": pd.to_datetime(last_closed.get("Date")).strftime("%Y-%m-%d %H:%M:%S")
        if pd.notna(last_closed.get("Date"))
        else "",
        "price": f"{current_price:.8f}",
        "signal": signal,
        "reason": reason,
        "position": position_state.get("position") or "",
        "entry_price": f"{position_state.get('entry_price', 0.0):.8f}",
        "stop_loss": f"{position_state.get('stop_loss', 0.0):.8f}",
        "take_profit": f"{position_state.get('take_profit', 0.0):.8f}",
        "capital": f"{capital:.2f}",
        "ema_fast": _fmt_float(last_closed.get("ema_fast")),
        "ema_medium": _fmt_float(last_closed.get("ema_medium")),
        "ema_slow": _fmt_float(last_closed.get("ema_slow")),
        "is_inside_bar": str(bool(last_closed.get("is_inside_bar", False))),
        "avg_deviation_pct": _fmt_float(last_closed.get("avg_deviation_pct")),
        "adx": _fmt_float(last_closed.get("adx")),
        "atr": _fmt_float(last_closed.get("atr")),
        "trend_bias": _fmt_float(last_closed.get("trend_bias")),
        # Campos adicionais derivados
        "close_last_closed": _fmt_float(c),
        "high_last_closed": _fmt_float(h),
        "low_last_closed": _fmt_float(l),
        "high_current": _fmt_float(cur_high),
        "low_current": _fmt_float(cur_low),
        "uptrend": str(bool(uptrend)),
        "downtrend": str(bool(downtrend)),
        "pullback_long_ok": str(bool(pullback_long_ok)),
        "pullback_short_ok": str(bool(pullback_short_ok)),
        "allow_long": str(bool(allow_long)),
        "allow_short": str(bool(allow_short)),
        "adx_ok": str(bool(adx_ok)),
        "atr_ok": str(bool(atr_ok)),
        "deviation_ok": str(bool(deviation_ok)),
        # Limiares atuais da configuração
        "adx_threshold": _fmt_float(params.get("adx_threshold")),
        "min_atr": _fmt_float(params.get("min_atr")),
        "max_avg_deviation_pct": _fmt_float(params.get("max_avg_deviation_pct")),
        # Quick audit flags
        "setup_ok": str(bool(setup_ok)),
        "entry_confirmed": str(bool(entry_breakout)),
        # Novos campos: preço de rompimento do candle anterior e flag offline
        "prev_high_for_entry": _fmt_float(h) if signal == "buy" else "",
        "prev_low_for_entry": _fmt_float(l) if signal == "sell" else "",
        # Não sabemos o 'would_enter_backtest' em tempo real; será calculado por script offline
        "would_enter_backtest": "",
    }
    # Append with csv module to handle commas safely
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writerow(row)


def _append_trade(
    event_time,
    event_type: str,
    subtype: str,
    side: str,
    price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    capital: float,
    reason: str = "",
) -> None:
    global TRADES_CSV_PATH
    if TRADES_CSV_PATH is None:
        return
    try:
        t = pd.to_datetime(event_time)
        if getattr(t, "tzinfo", None) is not None:
            t = t.tz_localize(None)
        event_time_str = t.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        event_time_str = ""
    row = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "event_time": event_time_str,
        "type": event_type,
        "subtype": subtype,
        "side": side,
        "price": f"{float(price):.8f}",
        "entry_price": f"{float(entry_price):.8f}",
        "stop_loss": f"{float(stop_loss):.8f}",
        "take_profit": f"{float(take_profit):.8f}",
        "capital": f"{float(capital):.2f}",
        "reason": reason,
    }
    with TRADES_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writerow(row)


def _fmt_float(x) -> str:
    try:
        if x is None or np.isnan(x):
            return ""
        return f"{float(x):.8f}"
    except Exception:
        return ""


def _render_png(
    df: pd.DataFrame,
    current_price: float,
    params: dict,
    png_path: Path,
    signal: str,
    reason: str,
    max_bars: int = 400,
) -> None:
    """Renders a compact chart with Close and EMAs, marking inside bars and SL/TP if any."""
    if df.empty:
        return
    # Use only closed bars to stay consistent with compute_signal
    n = len(df)
    if n < 2:
        return
    end_idx = n - 1  # exclude last forming candle
    start_idx = max(0, end_idx - max_bars)
    plot_df = df.iloc[start_idx:end_idx].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = plot_df["Date"]
    ax.plot(x, plot_df["close"], color="#1f77b4", linewidth=1.2, label="close")
    ax.plot(x, plot_df.get("ema_fast"), color="#ff7f0e", linewidth=1.0, label=f"EMA{params['ema_fast_period']}")
    ax.plot(x, plot_df.get("ema_medium"), color="#2ca02c", linewidth=1.0, label=f"EMA{params['ema_medium_period']}")
    ax.plot(x, plot_df.get("ema_slow"), color="#d62728", linewidth=1.0, label=f"EMA{params['ema_slow_period']}")

    # Mark inside bars on their close
    ib = plot_df[plot_df.get("is_inside_bar", False) == True]
    if not ib.empty:
        ax.scatter(ib["Date"], ib["close"], s=10, color="#9467bd", label="inside bar")

    # Draw levels if in position
    if position_state.get("position"):
        ep = position_state.get("entry_price", 0.0)
        sl = position_state.get("stop_loss", 0.0)
        tp = position_state.get("take_profit", 0.0)
        if ep:
            ax.axhline(ep, color="#8c564b", linestyle="--", linewidth=1.0, label="entry")
        if sl:
            ax.axhline(sl, color="#e377c2", linestyle=":", linewidth=1.0, label="stop")
        if tp:
            ax.axhline(tp, color="#17becf", linestyle=":", linewidth=1.0, label="target")

    # Plot trade event markers (entries/exits) within the visible window
    t0 = plot_df["Date"].iloc[0]
    t1 = plot_df["Date"].iloc[-1]
    entries_long_x, entries_long_y = [], []
    entries_short_x, entries_short_y = [], []
    exits_long_x, exits_long_y = [], []
    exits_short_x, exits_short_y = [], []
    # Rastreia o último P&L realizado (para título/anotação)
    last_exit_xy = None
    last_exit_pnl = None
    last_exit_cap = None
    for ev in TRADE_EVENTS:
        try:
            ev_time = pd.to_datetime(ev.get("time")).to_pydatetime()
        except Exception:
            continue
        # Ensure naive
        if getattr(ev_time, "tzinfo", None) is not None:
            ev_time = ev_time.replace(tzinfo=None)
        if not (t0 <= ev_time <= t1):
            continue
        if ev.get("type") == "entry":
            if ev.get("side") == "long":
                entries_long_x.append(ev_time)
                entries_long_y.append(ev.get("price"))
            else:
                entries_short_x.append(ev_time)
                entries_short_y.append(ev.get("price"))
        else:
            if ev.get("side") == "long":
                exits_long_x.append(ev_time)
                price_val = ev.get("price")
                exits_long_y.append(price_val)
                last_exit_xy = (ev_time, price_val)
                last_exit_pnl = ev.get("pnl") if ev.get("pnl") is not None else last_exit_pnl
                last_exit_cap = ev.get("capital") if ev.get("capital") is not None else last_exit_cap
            else:
                exits_short_x.append(ev_time)
                price_val = ev.get("price")
                exits_short_y.append(price_val)
                last_exit_xy = (ev_time, price_val)
                last_exit_pnl = ev.get("pnl") if ev.get("pnl") is not None else last_exit_pnl
                last_exit_cap = ev.get("capital") if ev.get("capital") is not None else last_exit_cap

    if entries_long_x:
        ax.scatter(entries_long_x, entries_long_y, marker="^", s=50, color="#2ca02c", edgecolor="black", linewidths=0.5, label="entry long")
    if entries_short_x:
        ax.scatter(entries_short_x, entries_short_y, marker="v", s=50, color="#d62728", edgecolor="black", linewidths=0.5, label="entry short")
    if exits_long_x:
        ax.scatter(exits_long_x, exits_long_y, marker="x", s=50, color="#2ca02c", linewidths=1.2, label="exit long")
    if exits_short_x:
        ax.scatter(exits_short_x, exits_short_y, marker="x", s=50, color="#d62728", linewidths=1.2, label="exit short")

    # Optionally show current price as well
    ax.axhline(current_price, color="#7f7f7f", linestyle="-.", linewidth=0.8, label="last price")

    # Anota o último P&L realizado, se disponível
    if last_exit_xy and (last_exit_pnl is not None):
        color_pnl = "#2ca02c" if float(last_exit_pnl) >= 0 else "#d62728"
        ax.annotate(
            f"{float(last_exit_pnl):+,.2f}",
            xy=last_exit_xy,
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=8,
            color=color_pnl,
            ha="center",
        )

    cap_now = position_state.get("capital", 0.0)
    extra = (
        f" | last PnL: {float(last_exit_pnl):+,.2f} | capital: ${float(last_exit_cap if last_exit_cap is not None else cap_now):,.2f}"
        if last_exit_pnl is not None
        else f" | capital: ${float(cap_now):,.2f}"
    )
    ax.set_title(
        f"ALBROOKS {params['ticker']}@{params['interval']} | {plot_df['Date'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"signal={signal} | {reason}{extra}"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close(fig)


def check_for_new_entry(df: pd.DataFrame, current_price: float, params: dict) -> None:
    """Verifica se há novo sinal de entrada e abre posição caso aplicável."""
    if position_state["position"]:
        return

    signal, reason = compute_signal(df, params)
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    if signal == "buy":
        entry_price = df.iloc[-2]["high"]
        # Confirm breakout using current candle high to avoid missing fast touches
        if df.iloc[-1]["high"] >= entry_price:
            stop, target = calculate_levels(df, params, "long", entry_price)
            if stop is None or target is None:
                print(f"[{now_str}] SINAL LONG descartado (níveis inválidos).")
                return
            slip_pct = float(params.get("slippage_pct", 0.0))
            entry_fill = float(entry_price) * (1 + slip_pct)
            position_state.update({
                "position": "long",
                "entry_price": entry_price,
                "stop_loss": stop,
                "take_profit": target,
                "entry_fill": entry_fill,
            })
            print(f"[{now_str}] ENTRADA LONG | Preço: {entry_price:.2f} | Stop: {stop:.2f} | Alvo: {target:.2f}")
            # Log entry marker at the time of the last closed candle
            entry_time = df.iloc[-2]["Date"]
            try:
                t = pd.to_datetime(entry_time)
                if getattr(t, "tzinfo", None) is not None:
                    t = t.tz_localize(None)
            except Exception:
                t = datetime.now().replace(tzinfo=None)
            # Usa o preço de execução (fill) para o marcador do gráfico
            TRADE_EVENTS.append({
                "type": "entry",
                "side": "long",
                "time": t,
                "price": float(entry_fill),
                "label": "ENTRY",
            })
            # Persist entry to trades CSV
            _append_trade(
                event_time=t,
                event_type="entry",
                subtype="ENTRY",
                side="long",
                # Persiste o preço de execução
                price=float(entry_fill),
                entry_price=float(entry_price),
                stop_loss=float(stop),
                take_profit=float(target),
                capital=float(position_state.get("capital", 0.0)),
                reason=reason,
            )
        else:
            print(f"[{now_str}] SINAL LONG detectado, aguardando rompimento de {entry_price:.2f}...")
    elif signal == "sell":
        entry_price = df.iloc[-2]["low"]
        # Confirm breakout using current candle low to avoid missing fast touches
        if df.iloc[-1]["low"] <= entry_price:
            stop, target = calculate_levels(df, params, "short", entry_price)
            if stop is None or target is None:
                print(f"[{now_str}] SINAL SHORT descartado (níveis inválidos).")
                return
            slip_pct = float(params.get("slippage_pct", 0.0))
            entry_fill = float(entry_price) * (1 - slip_pct)
            position_state.update({
                "position": "short",
                "entry_price": entry_price,
                "stop_loss": stop,
                "take_profit": target,
                "entry_fill": entry_fill,
            })
            print(f"[{now_str}] ENTRADA SHORT | Preço: {entry_price:.2f} | Stop: {stop:.2f} | Alvo: {target:.2f}")
            entry_time = df.iloc[-2]["Date"]
            try:
                t = pd.to_datetime(entry_time)
                if getattr(t, "tzinfo", None) is not None:
                    t = t.tz_localize(None)
            except Exception:
                t = datetime.now().replace(tzinfo=None)
            TRADE_EVENTS.append({
                "type": "entry",
                "side": "short",
                "time": t,
                "price": float(entry_fill),
                "label": "ENTRY",
            })
            _append_trade(
                event_time=t,
                event_type="entry",
                subtype="ENTRY",
                side="short",
                price=float(entry_fill),
                entry_price=float(entry_price),
                stop_loss=float(stop),
                take_profit=float(target),
                capital=float(position_state.get("capital", 0.0)),
                reason=reason,
            )
        else:
            print(f"[{now_str}] SINAL SHORT detectado, aguardando rompimento de {entry_price:.2f}...")
    else:
        print(
            f"[{now_str}] SINAL: hold ({reason}) | PREÇO: {current_price:.2f} | CAPITAL: ${position_state['capital']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa a estratégia Al Brooks em modo 'live' (paper trading).")
    parser.add_argument("--ticker", default="BTCUSDT", help="Símbolo do ativo")
    parser.add_argument("--poll-interval", type=int, default=10, help="Intervalo de verificação em segundos")
    parser.add_argument("--capital", type=float, default=100.0, help="Capital inicial para o paper trading")
    args = parser.parse_args()

    # Intervalo fixo para este modo live
    interval = "1m"

    position_state["capital"] = args.capital
    position_state.update({"position": None, "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0})

    active_cfg = load_active_config(args.ticker, interval)
    if not active_cfg:
        print(f"ERRO: Nenhuma configuração ativa encontrada para {args.ticker}@{interval}.")
        print("Execute a otimização antes de iniciar o modo live.")
        sys.exit(1)

    # Usar asdict para consistência com os outros módulos
    from dataclasses import asdict

    params = asdict(active_cfg)
    print("--- Al Brooks Live Monitor ---")
    print(f"Configuração ativa para {args.ticker}@{interval}")
    print({k: v for k, v in params.items() if k not in {"ticker", "interval", "days"}})

    # Prepare live outputs (CSV overwritten on each run, PNG overwritten on each cycle)
    csv_path, png_path = _ensure_live_paths(args.ticker, interval)
    _init_csv(csv_path)
    # Prepare trades CSV (append-only)
    global TRADES_CSV_PATH
    TRADES_CSV_PATH = _ensure_trades_csv(args.ticker, interval)

    interval_minutes = 1

    candles_per_day = max(1, (24 * 60) // interval_minutes)

    print("\nIniciando monitoramento... pressione Ctrl+C para encerrar.")
    try:
        while True:
            days_needed = int((params["ema_slow_period"] * 2) / candles_per_day) + 2
            start_dt = datetime.now(UTC) - timedelta(days=days_needed)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

            df = get_historical_klines(args.ticker, interval, start_str)
            if df.empty or len(df) < params["ema_slow_period"]:
                print("Aguardando dados suficientes...")
                time.sleep(args.poll_interval)
                continue

            df = add_indicators(df, params)
            # Busca preço atual com fallback para o close do candle corrente
            try:
                current_price = get_current_price(args.ticker)
            except Exception as e:
                print(f"[warn] Falha ao buscar preço atual ({e}); usando close do candle.")
                try:
                    current_price = float(df.iloc[-1]["close"])
                except Exception:
                    # Como último recurso, usa o close do último fechado
                    current_price = float(df.iloc[-2]["close"]) if len(df) >= 2 else float("nan")

            manage_existing_position(df, current_price, params)
            check_for_new_entry(df, current_price, params)

            # Audit snapshot: compute signal, append CSV, render PNG
            signal, reason = compute_signal(df, params)
            last_closed = df.iloc[-2]
            cur_high = df.iloc[-1].get("high")
            cur_low = df.iloc[-1].get("low")
            _append_csv(
                csv_path,
                last_closed,
                current_price,
                signal,
                reason,
                position_state["capital"],
                cur_high,
                cur_low,
                params,
            )
            _render_png(df, current_price, params, png_path, signal, reason)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nMonitoramento encerrado pelo usuário.")


if __name__ == "__main__":
    main()
