from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, UTC

import numpy as np
import pandas as pd

from ...binance_client import get_current_price, get_historical_klines
from .config import load_active_config
from .indicators import add_indicators

position_state = {"position": None, "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "capital": 100.0}

PULLBACK_LOOKBACK = 10


def compute_signal(df: pd.DataFrame, params: dict) -> str:
    """Determina se há sinal de compra ou venda no último candle fechado."""
    if len(df) < 3:
        return "hold"

    last = df.iloc[-2]  # último candle fechado

    if last["avg_deviation_pct"] > params["max_avg_deviation_pct"]:
        return "hold"

    if np.isnan(last["atr"]) or last["atr"] <= params.get("min_atr", 0.0):
        return "hold"

    if np.isnan(last["adx"]) or last["adx"] < params["adx_threshold"]:
        return "hold"

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
        return "hold"

    if allow_long and uptrend and last["close"] < last["ema_fast"]:
        return "buy"

    if allow_short and downtrend and last["close"] > last["ema_fast"]:
        return "sell"

    return "hold"


def calculate_levels(df: pd.DataFrame, params: dict, direction: str, entry_price: float) -> tuple[float, float] | tuple[None, None]:
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


def handle_exit(exit_type: str, price: float, params: dict) -> None:
    """Fecha a posição atual e atualiza capital."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    entry_price = position_state["entry_price"]
    lot_size = params["lot_size"]

    if position_state["position"] == "long":
        pnl = (price - entry_price) * lot_size
    else:
        pnl = (entry_price - price) * lot_size

    position_state["capital"] += pnl
    print(f"[{now_str}] PREÇO: {price:.2f} | SAÍDA: {exit_type} | P&L: ${pnl:.2f} | CAPITAL: ${position_state['capital']:.2f}")
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
            handle_exit("STOP LOSS", position_state["stop_loss"], params)
        elif current_price >= position_state["take_profit"]:
            handle_exit("TAKE PROFIT", position_state["take_profit"], params)
        else:
            unrealized = (current_price - position_state["entry_price"]) * params["lot_size"]
            print(
                f"PREÇO: {current_price:.2f} | POSIÇÃO: LONG | STOP: {position_state['stop_loss']:.2f} | "
                f"ALVO: {position_state['take_profit']:.2f} | P&L flutuante: ${unrealized:.2f}"
            )
    else:
        if current_price >= position_state["stop_loss"]:
            handle_exit("STOP LOSS", position_state["stop_loss"], params)
        elif current_price <= position_state["take_profit"]:
            handle_exit("TAKE PROFIT", position_state["take_profit"], params)
        else:
            unrealized = (position_state["entry_price"] - current_price) * params["lot_size"]
            print(
                f"PREÇO: {current_price:.2f} | POSIÇÃO: SHORT | STOP: {position_state['stop_loss']:.2f} | "
                f"ALVO: {position_state['take_profit']:.2f} | P&L flutuante: ${unrealized:.2f}"
            )


def check_for_new_entry(df: pd.DataFrame, current_price: float, params: dict) -> None:
    """Verifica se há novo sinal de entrada e abre posição caso aplicável."""
    if position_state["position"]:
        return

    signal = compute_signal(df, params)
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    if signal == "buy":
        entry_price = df.iloc[-2]["high"]
        stop, target = calculate_levels(df, params, "long", entry_price)
        if stop is None or target is None:
            print(f"[{now_str}] SINAL LONG descartado (níveis inválidos).")
            return
        position_state.update({"position": "long", "entry_price": entry_price, "stop_loss": stop, "take_profit": target})
        print(f"[{now_str}] SINAL LONG | Entrada: {entry_price:.2f} | Stop: {stop:.2f} | Alvo: {target:.2f}")
    elif signal == "sell":
        entry_price = df.iloc[-2]["low"]
        stop, target = calculate_levels(df, params, "short", entry_price)
        if stop is None or target is None:
            print(f"[{now_str}] SINAL SHORT descartado (níveis inválidos).")
            return
        position_state.update({"position": "short", "entry_price": entry_price, "stop_loss": stop, "take_profit": target})
        print(f"[{now_str}] SINAL SHORT | Entrada: {entry_price:.2f} | Stop: {stop:.2f} | Alvo: {target:.2f}")
    else:
        print(f"[{now_str}] SINAL: hold | PREÇO: {current_price:.2f} | CAPITAL: ${position_state['capital']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa a estratégia Al Brooks em modo 'live' (paper trading).")
    parser.add_argument("--ticker", default="BTCUSDT", help="Símbolo do ativo")
    parser.add_argument("--interval", default="5m", help="Intervalo das velas")
    parser.add_argument("--poll-interval", type=int, default=10, help="Intervalo de verificação em segundos")
    parser.add_argument("--capital", type=float, default=100.0, help="Capital inicial para o paper trading")
    args = parser.parse_args()

    position_state["capital"] = args.capital
    position_state.update({"position": None, "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0})

    active_cfg = load_active_config(args.ticker, args.interval)
    if not active_cfg:
        print(f"ERRO: Nenhuma configuração ativa encontrada para {args.ticker}@{args.interval}.")
        print("Execute a otimização antes de iniciar o modo live.")
        sys.exit(1)

    params = active_cfg.to_dict()
    print("--- Al Brooks Live Monitor ---")
    print(f"Configuração ativa para {args.ticker}@{args.interval}")
    print({k: v for k, v in params.items() if k not in {"ticker", "interval", "days"}})

    interval_minutes = 1
    if args.interval.endswith("m"):
        interval_minutes = int(args.interval[:-1])
    elif args.interval.endswith("h"):
        interval_minutes = int(args.interval[:-1]) * 60
    elif args.interval.endswith("d"):
        interval_minutes = int(args.interval[:-1]) * 24 * 60

    candles_per_day = max(1, (24 * 60) // interval_minutes)

    print("\nIniciando monitoramento... pressione Ctrl+C para encerrar.")
    try:
        while True:
            days_needed = int((params["ema_slow_period"] * 2) / candles_per_day) + 2
            start_dt = datetime.now(UTC) - timedelta(days=days_needed)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

            df = get_historical_klines(args.ticker, args.interval, start_str)
            if df.empty or len(df) < params["ema_slow_period"]:
                print("Aguardando dados suficientes...")
                time.sleep(args.poll_interval)
                continue

            df = add_indicators(df, params)
            current_price = get_current_price(args.ticker)

            manage_existing_position(df, current_price, params)
            check_for_new_entry(df, current_price, params)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nMonitoramento encerrado pelo usuário.")


if __name__ == "__main__":
    main()
