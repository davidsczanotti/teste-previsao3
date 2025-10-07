from __future__ import annotations

import sys
import argparse
import time
from datetime import datetime, timedelta, UTC

import pandas as pd

from ...binance_client import get_current_price, get_historical_klines
from .config import load_active_config
from .indicators import add_indicators

# Estado da posição e capital. Usar um dicionário é uma forma simples de gerenciar o estado.
position_state = {"position": None, "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "capital": 100.0}


def check_signals(df: pd.DataFrame, params: dict) -> str:
    """
    Verifica os sinais de compra ou venda no último candle disponível.
    Retorna 'buy', 'sell', ou 'hold'.
    """
    # 1. Calcular Indicadores
    df["ema_fast"] = ta.ema(df["close"], length=params["ema_fast_period"])
    df["ema_medium"] = ta.ema(df["close"], length=params["ema_medium_period"])
    df["ema_slow"] = ta.ema(df["close"], length=params["ema_slow_period"]) df["is_inside_bar"] = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1)) df["avg_deviation_pct"] = abs((df["close"] - df["ema_slow"]) / df["ema_slow"]) * 100 # Pega os dados do penúltimo candle (o último candle fechado) last = df.iloc[-2]

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


def get_interval_in_minutes(interval_str: str) -> int:
    """Converte o string do intervalo (ex: '5m', '1h') para minutos."""
    if "m" in interval_str:
        return int(interval_str.replace("m", ""))
    if "h" in interval_str:
        return int(interval_str.replace("h", "")) * 60
    if "d" in interval_str:
        return int(interval_str.replace("d", "")) * 24 * 60
    return 1


def handle_exit(exit_type: str, current_price: float, params: dict):
    """Lida com a lógica de saída de uma posição (TP ou SL)."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    entry_price = position_state["entry_price"]
    lot_size = params["lot_size"]

    if position_state["position"] == "long":
        pnl = (current_price - entry_price) * lot_size
    else:  # short
        pnl = (entry_price - current_price) * lot_size

    position_state["capital"] += pnl
    print(
        f"[{now_str}] PREÇO: {current_price:.2f} | SAÍDA: *** {exit_type} *** | P&L: ${pnl:.2f} | CAPITAL: ${position_state['capital']:.2f}"
    )
    position_state["position"] = None


def manage_existing_position(current_price: float, params: dict):
    """Verifica e gerencia uma posição aberta (long ou short)."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    position = position_state["position"]
    entry_price = position_state["entry_price"]
    take_profit = position_state["take_profit"]
    stop_loss = position_state["stop_loss"]
    lot_size = params["lot_size"]

    if position == "long":
        if current_price >= take_profit:
            handle_exit("TAKE PROFIT (ALVO)", take_profit, params)
        elif current_price <= stop_loss:
            handle_exit("STOP LOSS (PERDA)", stop_loss, params)
        else:
            unrealized_pnl = (current_price - entry_price) * lot_size
            print(
                f"[{now_str}] PREÇO: {current_price:.2f} | POSIÇÃO: LONG | P&L flutuante: ${unrealized_pnl:.2f} | CAPITAL: ${position_state['capital']:.2f}"
            )
    elif position == "short":
        if current_price <= take_profit:
            handle_exit("TAKE PROFIT (ALVO)", take_profit, params)
        elif current_price >= stop_loss:
            handle_exit("STOP LOSS (PERDA)", stop_loss, params)
        else:
            unrealized_pnl = (entry_price - current_price) * lot_size
            print(
                f"[{now_str}] PREÇO: {current_price:.2f} | POSIÇÃO: SHORT | P&L flutuante: ${unrealized_pnl:.2f} | CAPITAL: ${position_state['capital']:.2f}"
            )


def check_for_new_entry(df: pd.DataFrame, current_price: float, params: dict):
    """Verifica por novos sinais de entrada e abre uma posição se aplicável."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    signal = check_signals(df.copy(), params)

    if signal == "buy":
        print(f"[{now_str}] PREÇO: {current_price:.2f} | SINAL: *** ENTRADA COMPRA (LONG) ***")
        position_state["position"] = "long"
        position_state["entry_price"] = current_price
        # Calcula SL/TP
        pullback_window = df.tail(10)
        stop_loss = pullback_window["low"].min()
        risk = max(0.0001 * current_price, current_price - stop_loss)  # Evita risco zero/negativo
        take_profit = current_price + (risk * params["risk_reward_ratio"])
        position_state["stop_loss"] = stop_loss
        position_state["take_profit"] = take_profit
        print(f" -> Alvo: {take_profit:.2f} | Stop: {stop_loss:.2f}")

    elif signal == "sell":
        print(f"[{now_str}] PREÇO: {current_price:.2f} | SINAL: *** ENTRADA VENDA (SHORT) ***")
        position_state["position"] = "short"
        position_state["entry_price"] = current_price
        # Calcula SL/TP
        rally_window = df.tail(10)
        stop_loss = rally_window["high"].max()
        risk = max(0.0001 * current_price, stop_loss - current_price)  # Evita risco zero/negativo
        take_profit = current_price - (risk * params["risk_reward_ratio"])
        position_state["stop_loss"] = stop_loss
        position_state["take_profit"] = take_profit
        print(f" -> Alvo: {take_profit:.2f} | Stop: {stop_loss:.2f}")
    else:
        print(f"[{now_str}] PREÇO: {current_price:.2f} | SINAL: hold | CAPITAL: ${position_state['capital']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Executa a estratégia Al Brooks em modo 'live' (paper trading).")
    parser.add_argument("--ticker", default="BTCUSDT", help="Símbolo do ativo")
    parser.add_argument("--interval", default="1m", help="Intervalo das velas")
    parser.add_argument("--poll_interval_seconds", type=int, default=10, help="Intervalo de verificação em segundos")
    parser.add_argument("--capital", type=float, default=100.0, help="Capital inicial para o paper trading")
    args = parser.parse_args()

    # Inicializa o estado da posição e capital
    position_state["capital"] = args.capital

    # Limpa o estado da posição para uma nova execução
    position_state.update({"position": None, "entry_price": 0.0, "stop_loss": 0.0, "take_profit": 0.0})

    # Tenta carregar a configuração ativa
    active_cfg = load_active_config(args.ticker, args.interval)
    if not active_cfg:
        print(f"ERRO: Nenhuma configuração ativa encontrada para {args.ticker}@{args.interval}.")
        print("Por favor, execute o script de otimização primeiro.")
        sys.exit(1)

    print("--- Al Brooks Live Signal Monitor ---")
    print(f"Usando configuração ativa para {args.ticker}@{args.interval}")
    params = active_cfg.to_dict()
    print("Parâmetros carregados:", {k: v for k, v in params.items() if k not in ["ticker", "interval", "days"]})

    interval_minutes = get_interval_in_minutes(args.interval)
    candles_per_day = (24 * 60) / interval_minutes

    # Loop principal
    print("\nIniciando monitoramento... Pressione Ctrl+C para parar.")
    while True:
        try:
            # Carrega um pouco mais de dados do que o necessário para o cálculo das EMAs
            # BUGFIX: O cálculo estava com '15' fixo. Agora usa o 'candles_per_day' correto.
            days_to_load = (
                int(params["ema_slow_period"] * 2 / candles_per_day) + 2
            )  # Aprox. dias para 2x a EMA lenta + margem
            start_dt = datetime.now(UTC) - timedelta(days=days_to_load)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

            df = get_historical_klines(args.ticker, args.interval, start_str)

            if df.empty or len(df) < params["ema_slow_period"]:
                print(
                    f"Aguardando dados suficientes... (necessário: {params['ema_slow_period']}, disponível: {len(df)})"
                )
                time.sleep(args.poll_interval_seconds)
                continue

            df = add_indicators(df, params)

            current_price = get_current_price(args.ticker)
            if position_state["position"]:
                manage_existing_position(current_price, params)
            else:
                check_for_new_entry(df, current_price, params)

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
