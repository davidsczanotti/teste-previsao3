from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, UTC

import pandas as pd
import pandas_ta as ta

from ...binance_client import get_historical_klines, get_current_price
from .config import load_active_config

position_state = {"position": None, "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0}


def check_signals(df: pd.DataFrame, params: dict) -> str:
    """
    Verifica os sinais de compra ou venda no último candle disponível.
    Retorna 'buy', 'sell', ou 'hold'.
    """
    # 1. Calcular Indicadores
    df["ema_fast"] = ta.ema(df["close"], length=params["ema_fast_period"])
    df["ema_medium"] = ta.ema(df["close"], length=params["ema_medium_period"])
    df["ema_slow"] = ta.ema(df["close"], length=params["ema_slow_period"])
    df["is_inside_bar"] = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    df["avg_deviation_pct"] = abs((df["close"] - df["ema_slow"]) / df["ema_slow"]) * 100

    # Pega os dados do penúltimo candle (o último candle fechado)
    last = df.iloc[-2]

    # Filtro de afastamento médio
    is_close_to_ema = last["avg_deviation_pct"] <= params["max_avg_deviation_pct"]

    # --- Condição de Compra (Buy Signal) ---
    is_uptrend = (
        last["close"] > last["ema_medium"]
        and last["ema_fast"] > last["ema_medium"]
        and last["ema_medium"] > last["ema_slow"]
    )
    is_pullback = last["close"] < last["ema_fast"]

    if last["is_inside_bar"] and is_uptrend and is_pullback and is_close_to_ema:
        return "buy"

    # --- Condição de Venda (Sell Signal) ---
    is_downtrend = (
        last["close"] < last["ema_medium"]
        and last["ema_fast"] < last["ema_medium"]
        and last["ema_medium"] < last["ema_slow"]
    )
    is_rally = last["close"] > last["ema_fast"]

    if last["is_inside_bar"] and is_downtrend and is_rally and is_close_to_ema:
        return "sell"

    return "hold"


def main():
    parser = argparse.ArgumentParser(description="Executa a estratégia Al Brooks em modo 'live' (paper trading).")
    parser.add_argument("--ticker", default="BTCUSDT", help="Símbolo do ativo")
    parser.add_argument("--interval", default="15m", help="Intervalo das velas")
    parser.add_argument("--poll_interval_seconds", type=int, default=10, help="Intervalo de verificação em segundos")
    args = parser.parse_args()

    print("--- Al Brooks Live Signal Monitor ---")

    # Tenta carregar a configuração ativa
    active_cfg = load_active_config(args.ticker, args.interval)

    if not active_cfg:
        print(f"ERRO: Nenhuma configuração ativa encontrada para {args.ticker}@{args.interval}.")
        print("Por favor, execute o script de otimização primeiro.")
        return

    print(f"Usando configuração ativa para {args.ticker}@{args.interval}")
    params = active_cfg.to_dict()
    print("Parâmetros carregados:", {k: v for k, v in params.items() if k not in ["ticker", "interval", "days"]})

    # Loop principal
    print("\nIniciando monitoramento... Pressione Ctrl+C para parar.")
    while True:
        try:
            # Carrega um pouco mais de dados do que o necessário para o cálculo das EMAs
            days_to_load = int(params["ema_slow_period"] * 2 / (24 * 60 / 15)) + 1  # Aprox. dias para 2x a EMA lenta
            start_dt = datetime.now(UTC) - timedelta(days=days_to_load)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

            df = get_historical_klines(args.ticker, args.interval, start_str)

            if df.empty or len(df) < params["ema_slow_period"]:
                print("Aguardando dados suficientes...")
                time.sleep(args.poll_interval_seconds)
                continue

            # Exibe o status
            current_price = get_current_price(args.ticker)
            now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

            # --- LÓGICA DE GESTÃO DE POSIÇÃO ---

            # 1. Se estiver em uma posição, verificar saída
            if position_state["position"] == "long":
                if current_price >= position_state["take_profit"]:
                    print(f"[{now_str}] PREÇO: {current_price:.2f} | SAÍDA: *** ATINGIU TAKE PROFIT (ALVO) ***")
                    position_state["position"] = None
                elif current_price <= position_state["stop_loss"]:
                    print(f"[{now_str}] PREÇO: {current_price:.2f} | SAÍDA: *** ATINGIU STOP LOSS (PERDA) ***")
                    position_state["position"] = None
                else:
                    pnl = (current_price - position_state["entry_price"]) * params["lot_size"]
                    print(f"[{now_str}] PREÇO: {current_price:.2f} | POSIÇÃO: LONG | P&L: ${pnl:.2f}")

            elif position_state["position"] == "short":
                if current_price <= position_state["take_profit"]:
                    print(f"[{now_str}] PREÇO: {current_price:.2f} | SAÍDA: *** ATINGIU TAKE PROFIT (ALVO) ***")
                    position_state["position"] = None
                elif current_price >= position_state["stop_loss"]:
                    print(f"[{now_str}] PREÇO: {current_price:.2f} | SAÍDA: *** ATINGIU STOP LOSS (PERDA) ***")
                    position_state["position"] = None
                else:
                    pnl = (position_state["entry_price"] - current_price) * params["lot_size"]
                    print(f"[{now_str}] PREÇO: {current_price:.2f} | POSIÇÃO: SHORT | P&L: ${pnl:.2f}")

            # 2. Se não estiver em uma posição, verificar entrada
            else:
                signal = check_signals(df.copy(), params)
                if signal == "buy":
                    print(f"[{now_str}] PREÇO: {current_price:.2f} | SINAL: *** ENTRADA COMPRA (LONG) ***")
                    position_state["position"] = "long"
                    position_state["entry_price"] = current_price
                    # Calcula SL/TP (lógica simplificada do backtest)
                    pullback_window = df.tail(10)
                    stop_loss = pullback_window["low"].min()
                    risk = current_price - stop_loss
                    take_profit = current_price + (risk * params["risk_reward_ratio"])
                    position_state["stop_loss"] = stop_loss
                    position_state["take_profit"] = take_profit
                    print(f" -> Alvo: {take_profit:.2f} | Stop: {stop_loss:.2f}")
                elif signal == "sell":
                    print(f"[{now_str}] PREÇO: {current_price:.2f} | SINAL: *** ENTRADA VENDA (SHORT) ***")
                    position_state["position"] = "short"
                    position_state["entry_price"] = current_price
                    # Calcula SL/TP (lógica simplificada do backtest)
                    rally_window = df.tail(10)
                    stop_loss = rally_window["high"].max()
                    risk = stop_loss - current_price
                    take_profit = current_price - (risk * params["risk_reward_ratio"])
                    position_state["stop_loss"] = stop_loss
                    position_state["take_profit"] = take_profit
                    print(f" -> Alvo: {take_profit:.2f} | Stop: {stop_loss:.2f}")
                else:
                    print(f"[{now_str}] PREÇO: {current_price:.2f} | SINAL: hold")

            # Aguarda o próximo ciclo
            time.sleep(args.poll_interval_seconds)

        except KeyboardInterrupt:
            print("\nEncerrando monitor...")
            break
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            print("Aguardando 60 segundos antes de tentar novamente...")
            time.sleep(60)


if __name__ == "__main__":
    main()
