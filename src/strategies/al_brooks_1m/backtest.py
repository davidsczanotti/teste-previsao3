from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, UTC

import mplfinance as mpf
import numpy as np
import pandas as pd
import pandas_ta as ta
from matplotlib import pyplot as plt

from ...binance_client import get_historical_klines
from .config import load_active_config


def backtest_al_brooks_inside_bar(
    df: pd.DataFrame,
    ema_fast_period: int = 10,
    ema_medium_period: int = 20,
    ema_slow_period: int = 50,
    risk_reward_ratio: float = 2.0,
    max_avg_deviation_pct: float = 0.5,  # Novo parâmetro: % máxima de afastamento da EMA 50
    lot_size: float = 0.1,
):
    """
    Executa um backtest para a estratégia de Inside Bar de Al Brooks.
    A lógica principal é a de "reversão" (compra no pullback em tendência de alta).
    """
    if df.empty:
        return [], 0.0, df

    # 1. Calcular Indicadores
    df["ema_fast"] = ta.ema(df["close"], length=ema_fast_period)
    df["ema_medium"] = ta.ema(df["close"], length=ema_medium_period)
    df["ema_slow"] = ta.ema(df["close"], length=ema_slow_period)

    # 2. Identificar Inside Bars
    # Um candle cuja máxima é menor que a do anterior e a mínima é maior que a do anterior.
    df["is_inside_bar"] = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))

    # 2.1. Calcular Afastamento Médio (em % da EMA Lenta)
    # abs((close - ema_slow) / ema_slow) * 100
    df["avg_deviation_pct"] = abs((df["close"] - df["ema_slow"]) / df["ema_slow"]) * 100

    trades = []
    position = None  # Pode ser 'long', 'short' ou None

    # 3. Iterar sobre os dados para simular as operações
    for i in range(1, len(df)):
        # --- Lógica de Saída ---
        if position == "long":
            # Verifica se atingiu o Take Profit ou Stop Loss
            if df["high"].iloc[i] >= trades[-1]["take_profit"]:
                trades[-1]["exit_price"] = trades[-1]["take_profit"]
                trades[-1]["exit_date"] = df["Date"].iloc[i]
                trades[-1]["pnl"] = (trades[-1]["exit_price"] - trades[-1]["entry_price"]) * lot_size
                position = None
            elif df["low"].iloc[i] <= trades[-1]["stop_loss"]:
                trades[-1]["exit_price"] = trades[-1]["stop_loss"]
                trades[-1]["exit_date"] = df["Date"].iloc[i]
                trades[-1]["pnl"] = (trades[-1]["exit_price"] - trades[-1]["entry_price"]) * lot_size
                position = None
        elif position == "short":
            # Verifica se atingiu o Take Profit ou Stop Loss para a venda
            if df["low"].iloc[i] <= trades[-1]["take_profit"]:
                trades[-1]["exit_price"] = trades[-1]["take_profit"]
                trades[-1]["exit_date"] = df["Date"].iloc[i]
                trades[-1]["pnl"] = (trades[-1]["entry_price"] - trades[-1]["exit_price"]) * lot_size
                position = None
            elif df["high"].iloc[i] >= trades[-1]["stop_loss"]:
                trades[-1]["exit_price"] = trades[-1]["stop_loss"]
                trades[-1]["exit_date"] = df["Date"].iloc[i]
                trades[-1]["pnl"] = (trades[-1]["entry_price"] - trades[-1]["exit_price"]) * lot_size
                position = None

        # --- Lógica de Entrada ---
        if position is None:
            # --- Condição de Compra (Buy Signal) ---
            is_uptrend = (
                df["close"].iloc[i - 1] > df["ema_medium"].iloc[i - 1]
                and df["ema_fast"].iloc[i - 1] > df["ema_medium"].iloc[i - 1]
                and df["ema_medium"].iloc[i - 1] > df["ema_slow"].iloc[i - 1]
            )
            is_pullback = df["close"].iloc[i - 1] < df["ema_fast"].iloc[i - 1]

            # Adicionar filtro de afastamento médio
            is_close_to_ema = df["avg_deviation_pct"].iloc[i - 1] <= max_avg_deviation_pct

            if df["is_inside_bar"].iloc[i - 1] and is_uptrend and is_pullback and is_close_to_ema:
                # O sinal é no candle anterior (i-1), a entrada é no candle atual (i)
                entry_price = df["high"].iloc[i - 1]  # Gatilho no rompimento da máxima

                # Verifica se o candle atual acionou a entrada
                if df["high"].iloc[i] > entry_price:
                    # Para o stop, procuramos a mínima do pullback recente (últimos 10 candles, por exemplo)
                    pullback_window = df.iloc[max(0, i - 10) : i]
                    stop_loss = pullback_window["low"].min()

                    risk = entry_price - stop_loss
                    if risk <= 0:
                        continue

                    take_profit = entry_price + (risk * risk_reward_ratio)

                    trades.append(
                        {
                            "entry_date": df["Date"].iloc[i],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "type": "long",
                        }
                    )
                    position = "long"

            # --- Condição de Venda (Sell Signal) ---
            is_downtrend = (
                df["close"].iloc[i - 1] < df["ema_medium"].iloc[i - 1]
                and df["ema_fast"].iloc[i - 1] < df["ema_medium"].iloc[i - 1]
                and df["ema_medium"].iloc[i - 1] < df["ema_slow"].iloc[i - 1]
            )
            is_rally = df["close"].iloc[i - 1] > df["ema_fast"].iloc[i - 1]

            if df["is_inside_bar"].iloc[i - 1] and is_downtrend and is_rally and is_close_to_ema:
                # O sinal é no candle anterior (i-1), a entrada é no candle atual (i)
                entry_price = df["low"].iloc[i - 1]  # Gatilho no rompimento da mínima

                # Verifica se o candle atual acionou a entrada
                if df["low"].iloc[i] < entry_price:
                    # Para o stop, procuramos a máxima do rally recente (últimos 10 candles)
                    rally_window = df.iloc[max(0, i - 10) : i]
                    stop_loss = rally_window["high"].max()

                    risk = stop_loss - entry_price
                    if risk <= 0:
                        continue

                    take_profit = entry_price - (risk * risk_reward_ratio)

                    trades.append(
                        {
                            "entry_date": df["Date"].iloc[i],
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "type": "short",
                        }
                    )
                    position = "short"

    # Calcular P&L total (considerando long e short)
    for trade in trades:
        if "pnl" not in trade:  # Se a operação não foi fechada
            trade["pnl"] = 0

    # Calcular P&L total
    total_pnl = sum(t.get("pnl", 0) for t in trades)

    return trades, total_pnl, df


def plot_backtest(df: pd.DataFrame, trades: list, ticker: str):
    """Plota o gráfico do backtest com os trades."""
    df_plot = df.set_index("Date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )

    # Limita para as últimas 500 velas para legibilidade
    if len(df_plot) > 500:
        df_plot = df_plot.tail(500)

    addplots = []

    # Adiciona EMAs ao gráfico
    addplots.append(mpf.make_addplot(df_plot["ema_fast"], color="blue"))
    addplots.append(mpf.make_addplot(df_plot["ema_medium"], color="orange"))
    addplots.append(mpf.make_addplot(df_plot["ema_slow"], color="purple"))

    # Adiciona marcadores de trade
    buy_markers = [np.nan] * len(df_plot)
    sell_markers = [np.nan] * len(df_plot)

    for trade in trades:
        if "entry_date" not in trade:
            continue
        try:
            idx = df_plot.index.get_indexer([trade["entry_date"]], method="nearest")[0]
            if trade["type"] == "long":
                buy_markers[idx] = df_plot["Low"].iloc[idx] * 0.98
            else:
                sell_markers[idx] = df_plot["High"].iloc[idx] * 1.02
        except KeyError:
            continue  # Ignora trades fora da janela de plotagem

    if any(not np.isnan(v) for v in buy_markers):
        addplots.append(mpf.make_addplot(buy_markers, type="scatter", marker="^", color="green", markersize=100))
    if any(not np.isnan(v) for v in sell_markers):
        addplots.append(mpf.make_addplot(sell_markers, type="scatter", marker="v", color="red", markersize=100))

    # Gera o gráfico
    chart_dir = "reports/charts"
    os.makedirs(chart_dir, exist_ok=True)
    filename = f"{chart_dir}/al_brooks_backtest_{ticker}.png"

    mpf.plot(
        df_plot,
        type="candle",
        style="yahoo",
        title=f"Al Brooks Inside Bar Backtest - {ticker}",
        addplot=addplots,
        savefig=filename,
    )
    print(f"\nGráfico do backtest salvo em: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Backtest para a estratégia de Inside Bar de Al Brooks.")
    parser.add_argument("--ticker", default="BTCUSDT", help="Símbolo do ativo (ex: BTCUSDT)")
    parser.add_argument("--interval", default="1m", help="Intervalo das velas (ex: 15m, 1h)")
    parser.add_argument("--days", type=int, default=365, help="Dias de dados históricos para o backtest")
    parser.add_argument("--lot_size", type=float, default=0.1, help="Tamanho do lote para cada operação")
    args = parser.parse_args()

    # Tenta carregar a configuração ativa
    active_cfg = load_active_config(args.ticker, args.interval)

    if active_cfg:
        print(f"Usando configuração ativa para {args.ticker}@{args.interval}")
        params = {
            "ema_fast_period": active_cfg.ema_fast_period,
            "ema_medium_period": active_cfg.ema_medium_period,
            "ema_slow_period": active_cfg.ema_slow_period,
            "risk_reward_ratio": active_cfg.risk_reward_ratio,
            "max_avg_deviation_pct": active_cfg.max_avg_deviation_pct,
            "lot_size": active_cfg.lot_size,
        }
        days_to_load = active_cfg.days
    else:
        print("Nenhuma configuração ativa encontrada. Usando parâmetros padrão.")
        params = {
            "ema_fast_period": 10,
            "ema_medium_period": 20,
            "ema_slow_period": 50,
            "risk_reward_ratio": 2.0,
            "max_avg_deviation_pct": 0.5,
            "lot_size": args.lot_size,
        }
        days_to_load = args.days

    # Carregar dados
    print(f"Carregando dados: {args.ticker} @ {args.interval} dos últimos {days_to_load} dias...")
    start_dt = datetime.now(UTC) - timedelta(days=days_to_load)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    df = get_historical_klines(args.ticker, args.interval, start_str)

    if df.empty:
        print("Nenhum dado foi retornado. Verifique o ticker, intervalo e a conexão.")
        return
    print(f"Total de {len(df)} candles carregados.")

    # Executar o backtest
    print("Executando backtest...")
    trades, total_pnl, df_with_indicators = backtest_al_brooks_inside_bar(df.copy(), **params)

    # Exibir resultados
    print("\n--- Resultados do Backtest ---")
    print(f"Período Analisado: {df['Date'].iloc[0]} a {df['Date'].iloc[-1]}")

    closed_trades = [t for t in trades if "pnl" in t]
    num_trades = len(closed_trades)

    if num_trades == 0:
        print("Nenhuma operação foi fechada no período.")
        return

    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] <= 0]

    win_rate = (len(wins) / num_trades) * 100 if num_trades > 0 else 0

    total_profit = sum(t["pnl"] for t in wins)
    total_loss = sum(t["pnl"] for t in losses)

    avg_win = total_profit / len(wins) if wins else 0
    avg_loss = abs(total_loss / len(losses)) if losses else 0

    profit_factor = total_profit / abs(total_loss) if total_loss != 0 else float("inf")

    print(f"Resultado Final (P&L): $ {total_pnl:.2f}")
    print(f"Total de Operações Fechadas: {num_trades}")
    print(f"Taxa de Acerto: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Média de Ganho: $ {avg_win:.2f}")
    print(f"Média de Perda: $ {avg_loss:.2f}")

    # Calcula e exibe a duração média dos trades
    if num_trades > 0:
        total_duration = sum(
            (trade["exit_date"] - trade["entry_date"] for trade in closed_trades),
            timedelta(0),
        )
        avg_duration = total_duration / num_trades
        print(f"Duração Média do Trade: {str(avg_duration).split('.')[0]}")


    # Opcional: Salvar trades em um arquivo para análise mais profunda
    # pd.DataFrame(closed_trades).to_csv("al_brooks_trades.csv", index=False)
    # print("\nTrades salvos em 'al_brooks_trades.csv'")

    # Plotar o resultado
    plot_backtest(df_with_indicators, trades, args.ticker)


if __name__ == "__main__":
    main()
