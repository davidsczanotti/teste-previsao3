from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, UTC

import mplfinance as mpf
import numpy as np
import pandas as pd
from ...binance_client import get_historical_klines
from .config import AlBrooksConfig, load_active_config
from .indicators import add_indicators


def backtest_al_brooks_inside_bar(
    df: pd.DataFrame,
    ema_fast_period: int = 10,
    ema_medium_period: int = 20,
    ema_slow_period: int = 50,
    risk_reward_ratio: float = 2.0,
    max_avg_deviation_pct: float = 0.5,
    lot_size: float = 0.1,
    adx_period: int = 14,
    adx_threshold: float = 22.0,
    atr_period: int = 14,
    atr_stop_multiplier: float = 1.5,
    atr_trail_multiplier: float = 0.5,
    htf_lookback: int = 20,
    use_htf_bias: bool = True,
    min_atr: float = 0.0,
    pullback_lookback: int = 10,
) -> tuple[list[dict], float, pd.DataFrame]:
    """
    Executa um backtest para a estratégia de Inside Bar de Al Brooks utilizando filtros de tendência e volatilidade.

    A lógica considera:
      - Confirmar contexto com EMAs em múltiplos prazos
      - Filtrar períodos de consolidação através do ADX
      - Controlar afastamento da EMA lenta
      - Dimensionar stop/target por ATR com possibilidade de trailing
      - Filtrar direção pelo viés do EMA lento em janela maior (slope)
    """
    if df.empty:
        return [], 0.0, df

    params = {
        "ema_fast_period": ema_fast_period,
        "ema_medium_period": ema_medium_period,
        "ema_slow_period": ema_slow_period,
        "adx_period": adx_period,
        "atr_period": atr_period,
        "htf_lookback": htf_lookback,
    }
    df = add_indicators(df, params)

    trades: list[dict] = []
    position: str | None = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if position:
            trade = trades[-1]

            if atr_trail_multiplier > 0 and not np.isnan(row["atr"]):
                if position == "long":
                    trailing_stop = row["close"] - (row["atr"] * atr_trail_multiplier)
                    trade["stop_loss"] = max(trade["stop_loss"], trailing_stop)
                else:
                    trailing_stop = row["close"] + (row["atr"] * atr_trail_multiplier)
                    trade["stop_loss"] = min(trade["stop_loss"], trailing_stop)

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
                trade["exit_price"] = exit_price
                trade["exit_date"] = row["Date"]
                if position == "long":
                    trade["pnl"] = (exit_price - trade["entry_price"]) * lot_size
                else:
                    trade["pnl"] = (trade["entry_price"] - exit_price) * lot_size
                trade["exit_reason"] = exit_reason
                position = None
                continue

        if position is not None:
            continue

        if prev["avg_deviation_pct"] > max_avg_deviation_pct:
            continue

        if np.isnan(prev["atr"]) or prev["atr"] <= min_atr:
            continue

        if np.isnan(prev["adx"]) or prev["adx"] < adx_threshold:
            continue

        allow_long = True
        allow_short = True
        if use_htf_bias:
            bias = prev.get("trend_bias")
            if not np.isnan(bias):
                allow_long = bias >= 0
                allow_short = bias <= 0

        if not prev["is_inside_bar"]:
            continue

        uptrend = (
            prev["close"] > prev["ema_medium"]
            and prev["ema_fast"] > prev["ema_medium"]
            and prev["ema_medium"] > prev["ema_slow"]
        )
        downtrend = (
            prev["close"] < prev["ema_medium"]
            and prev["ema_fast"] < prev["ema_medium"]
            and prev["ema_medium"] < prev["ema_slow"]
        )
        pullback = prev["close"] < prev["ema_fast"]
        rally = prev["close"] > prev["ema_fast"]

        atr_value = prev["atr"]
        if np.isnan(atr_value) or atr_value <= 0:
            continue

        if allow_long and uptrend and pullback:
            entry_price = prev["high"]
            if row["high"] >= entry_price:
                lookback_slice = df.iloc[max(0, i - pullback_lookback) : i]
                pullback_low = lookback_slice["low"].min()
                stop_candidates = [entry_price - (atr_value * atr_stop_multiplier), pullback_low]
                stop_loss = min(stop_candidates)
                risk = entry_price - stop_loss
                if risk <= 0:
                    continue
                take_profit = entry_price + (risk * risk_reward_ratio)
                trades.append(
                    {
                        "entry_date": row["Date"],
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "type": "long",
                        "atr": atr_value,
                        "initial_risk": risk,
                    }
                )
                position = "long"
                continue

        if allow_short and downtrend and rally:
            entry_price = prev["low"]
            if row["low"] <= entry_price:
                lookback_slice = df.iloc[max(0, i - pullback_lookback) : i]
                rally_high = lookback_slice["high"].max()
                stop_candidates = [entry_price + (atr_value * atr_stop_multiplier), rally_high]
                stop_loss = max(stop_candidates)
                risk = stop_loss - entry_price
                if risk <= 0:
                    continue
                take_profit = entry_price - (risk * risk_reward_ratio)
                trades.append(
                    {
                        "entry_date": row["Date"],
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "type": "short",
                        "atr": atr_value,
                        "initial_risk": risk,
                    }
                )
                position = "short"
                continue

    if trades and position:
        trade = trades[-1]
        final_price = df["close"].iloc[-1]
        trade["exit_price"] = final_price
        trade["exit_date"] = df["Date"].iloc[-1]
        if position == "long":
            trade["pnl"] = (final_price - trade["entry_price"]) * lot_size
        else:
            trade["pnl"] = (trade["entry_price"] - final_price) * lot_size
        trade["exit_reason"] = "eod"
        position = None

    for trade in trades:
        trade.setdefault("pnl", 0.0)

    total_pnl = sum(trade["pnl"] for trade in trades)
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
    parser.add_argument("--interval", default="5m", help="Intervalo das velas (ex: 15m, 1h)")
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
    # BUGFIX: Apenas trades que realmente fecharam (têm 'exit_date') devem ser considerados.
    fully_closed_trades = [t for t in closed_trades if "exit_date" in t]
    if fully_closed_trades:
        total_duration = sum(
            (trade["exit_date"] - trade["entry_date"] for trade in fully_closed_trades),
            timedelta(0),
        )
        avg_duration = total_duration / len(fully_closed_trades)
        print(f"Duração Média do Trade: {str(avg_duration).split('.')[0]}")

    # Opcional: Salvar trades em um arquivo para análise mais profunda
    # pd.DataFrame(closed_trades).to_csv("al_brooks_trades.csv", index=False)
    # print("\nTrades salvos em 'al_brooks_trades.csv'")

    # Plotar o resultado
    plot_backtest(df_with_indicators, trades, args.ticker)


if __name__ == "__main__":
    main()
