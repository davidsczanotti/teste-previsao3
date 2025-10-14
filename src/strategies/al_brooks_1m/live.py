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

position_state = {"position": None, "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "capital": 100.0}

PULLBACK_LOOKBACK = 10

# In-memory trade events for plotting entries/exits during this session
TRADE_EVENTS: list[dict] = []


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

    if position_state["position"] == "long":
        pnl = (price - entry_price) * lot_size
    else:
        pnl = (entry_price - price) * lot_size

    position_state["capital"] += pnl
    print(
        f"[{now_str}] PREÇO: {price:.2f} | SAÍDA: {exit_type} | P&L: ${pnl:.2f} | CAPITAL: ${position_state['capital']:.2f}"
    )
    # Log exit marker for plotting
    try:
        t = pd.to_datetime(event_time)
        # Ensure naive datetime to align with DF axis
        if getattr(t, "tzinfo", None) is not None:
            t = t.tz_localize(None)
    except Exception:
        t = datetime.now().replace(tzinfo=None)
    TRADE_EVENTS.append({
        "type": "exit",
        "side": position_state.get("position") or "",
        "time": t,
        "price": float(price),
        "label": exit_type,
    })
    position_state.update({"position": None, "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0})


def manage_existing_position(df: pd.DataFrame, current_price: float, params: dict) -> None:
    """Atualiza posição aberta, aplicando trailing stop e verificando saídas."""
    position = position_state["position"]
    if not position:
        return

    last = df.iloc[-1]
    atr_value = last.get("atr", np.nan)
    trail_mult = params.get("atr_trail_multiplier", 0.0)

    if trail_mult > 0 and not np.isnan(atr_value):
        if position == "long":
            trailing = current_price - atr_value * trail_mult
            position_state["stop_loss"] = max(position_state["stop_loss"], trailing)
        else:
            trailing = current_price + atr_value * trail_mult
            position_state["stop_loss"] = min(position_state["stop_loss"], trailing)

    if position == "long":
        if current_price <= position_state["stop_loss"]:
            handle_exit("STOP LOSS", position_state["stop_loss"], params, df.iloc[-1]["Date"])
        elif current_price >= position_state["take_profit"]:
            handle_exit("TAKE PROFIT", position_state["take_profit"], params, df.iloc[-1]["Date"])
        else:
            unrealized = (current_price - position_state["entry_price"]) * params["lot_size"]
            print(
                f"PREÇO: {current_price:.2f} | POSIÇÃO: LONG | STOP: {position_state['stop_loss']:.2f} | "
                f"ALVO: {position_state['take_profit']:.2f} | P&L flutuante: ${unrealized:.2f}"
            )
    else:
        if current_price >= position_state["stop_loss"]:
            handle_exit("STOP LOSS", position_state["stop_loss"], params, df.iloc[-1]["Date"])
        elif current_price <= position_state["take_profit"]:
            handle_exit("TAKE PROFIT", position_state["take_profit"], params, df.iloc[-1]["Date"])
        else:
            unrealized = (position_state["entry_price"] - current_price) * params["lot_size"]
            print(
                f"PREÇO: {current_price:.2f} | POSIÇÃO: SHORT | STOP: {position_state['stop_loss']:.2f} | "
                f"ALVO: {position_state['take_profit']:.2f} | P&L flutuante: ${unrealized:.2f}"
            )


def _ensure_live_paths(ticker: str, interval: str) -> tuple[Path, Path]:
    """Ensures reports/live exists and returns (csv_path, png_path). Overwrites CSV header at start."""
    live_dir = Path("reports") / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ALBROOKS_{ticker}_{interval}"
    csv_path = live_dir / f"{stem}.csv"
    png_path = live_dir / f"{stem}.png"
    return csv_path, png_path


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
    ]
    csv_path.write_text(",".join(header) + "\n", encoding="utf-8")


def _append_csv(
    csv_path: Path,
    last_closed: pd.Series,
    current_price: float,
    signal: str,
    reason: str,
    capital: float,
    params: dict,
) -> None:
    # Deriva campos adicionais conforme a lógica de compute_signal
    c = last_closed.get("close")
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
    }
    # Append with csv module to handle commas safely
    with csv_path.open("a", newline="", encoding="utf-8") as f:
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
                exits_long_y.append(ev.get("price"))
            else:
                exits_short_x.append(ev_time)
                exits_short_y.append(ev.get("price"))

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

    ax.set_title(
        f"ALBROOKS {params['ticker']}@{params['interval']} | {plot_df['Date'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"signal={signal} | {reason}"
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
        if current_price >= entry_price:
            stop, target = calculate_levels(df, params, "long", entry_price)
            if stop is None or target is None:
                print(f"[{now_str}] SINAL LONG descartado (níveis inválidos).")
                return
            position_state.update(
                {"position": "long", "entry_price": entry_price, "stop_loss": stop, "take_profit": target}
            )
            print(f"[{now_str}] ENTRADA LONG | Preço: {entry_price:.2f} | Stop: {stop:.2f} | Alvo: {target:.2f}")
            # Log entry marker at the time of the last closed candle
            entry_time = df.iloc[-2]["Date"]
            try:
                t = pd.to_datetime(entry_time)
                if getattr(t, "tzinfo", None) is not None:
                    t = t.tz_localize(None)
            except Exception:
                t = datetime.now().replace(tzinfo=None)
            TRADE_EVENTS.append({"type": "entry", "side": "long", "time": t, "price": float(entry_price), "label": "ENTRY"})
        else:
            print(f"[{now_str}] SINAL LONG detectado, aguardando rompimento de {entry_price:.2f}...")
    elif signal == "sell":
        entry_price = df.iloc[-2]["low"]
        if current_price <= entry_price:
            stop, target = calculate_levels(df, params, "short", entry_price)
            if stop is None or target is None:
                print(f"[{now_str}] SINAL SHORT descartado (níveis inválidos).")
                return
            position_state.update(
                {"position": "short", "entry_price": entry_price, "stop_loss": stop, "take_profit": target}
            )
            print(f"[{now_str}] ENTRADA SHORT | Preço: {entry_price:.2f} | Stop: {stop:.2f} | Alvo: {target:.2f}")
            entry_time = df.iloc[-2]["Date"]
            try:
                t = pd.to_datetime(entry_time)
                if getattr(t, "tzinfo", None) is not None:
                    t = t.tz_localize(None)
            except Exception:
                t = datetime.now().replace(tzinfo=None)
            TRADE_EVENTS.append({"type": "entry", "side": "short", "time": t, "price": float(entry_price), "label": "ENTRY"})
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
            current_price = get_current_price(args.ticker)

            manage_existing_position(df, current_price, params)
            check_for_new_entry(df, current_price, params)

            # Audit snapshot: compute signal, append CSV, render PNG
            signal, reason = compute_signal(df, params)
            last_closed = df.iloc[-2]
            _append_csv(csv_path, last_closed, current_price, signal, reason, position_state["capital"], params)
            _render_png(df, current_price, params, png_path, signal, reason)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nMonitoramento encerrado pelo usuário.")


if __name__ == "__main__":
    main()
