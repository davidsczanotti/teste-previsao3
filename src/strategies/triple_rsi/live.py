import pandas as pd
import pytz
from datetime import datetime, timedelta, UTC
import time
from ...binance_client import get_historical_klines
from .config import load_active_config
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
import numpy as np


def live_triple_rsi(
    ticker="BTCUSDT",
    initial_capital=10000,
    lot_size=1,
    short_period=7,
    med_period=14,
    long_period=21,
    buy_entry=30,
    sell_entry=70,
    buy_exit=70,
    sell_exit=30,
    interval_minutes=5,
    invert=False,
    timezone="America/Sao_Paulo",
    history_days: int | None = None,
):
    # Define o fuso horário para exibição
    local_tz = pytz.timezone(timezone)

    def calculate_rsi(prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def plot_live_status(df_plot, rsi_s, rsi_m, rsi_l, trade_history, latest_time):
        """Gera e salva um gráfico do estado atual."""
        # Preparar dados para mplfinance (precisa de Open, High, Low, Close, Volume)
        # e um índice Datetime
        df_chart = df_plot.copy()
        df_chart = df_chart.set_index("Date").sort_index()
        # Renomeia as colunas para o padrão que mplfinance espera (primeira letra maiúscula)
        df_chart = df_chart.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )

        # Adicionar os RSIs ao DataFrame para plotagem, alinhando por posição
        # (o índice de df_chart é DatetimeIndex, enquanto os RSIs têm RangeIndex)
        df_chart["RSI_short"] = pd.Series(rsi_s.to_numpy(), index=df_chart.index)
        df_chart["RSI_med"] = pd.Series(rsi_m.to_numpy(), index=df_chart.index)
        df_chart["RSI_long"] = pd.Series(rsi_l.to_numpy(), index=df_chart.index)

        # Limita a quantidade de barras plotadas para manter legibilidade
        # Usa uma janela dinâmica baseada no período (3 × long_period)
        max_bars_to_plot = max(3 * long_period, 200)
        if len(df_chart) > max_bars_to_plot:
            df_chart = df_chart.tail(max_bars_to_plot)

        # Configurar os painéis de plotagem
        rsi_plots = [
            mpf.make_addplot(df_chart["RSI_short"], panel=1, color="green", ylabel="RSI", ylim=(0, 100)),
            mpf.make_addplot(df_chart["RSI_med"], panel=1, color="orange"),
            mpf.make_addplot(df_chart["RSI_long"], panel=1, color="red"),
        ]

        # Linhas de gatilho (entrada/saída) no painel de RSI para melhor leitura
        be_line = pd.Series(buy_entry, index=df_chart.index)
        se_line = pd.Series(sell_entry, index=df_chart.index)
        bx_line = pd.Series(buy_exit, index=df_chart.index)
        sx_line = pd.Series(sell_exit, index=df_chart.index)
        rsi_plots.extend(
            [
                mpf.make_addplot(be_line, panel=1, color="g", linestyle="--"),  # Buy Entry (ou Sell Entry invertido)
                mpf.make_addplot(se_line, panel=1, color="r", linestyle="--"),  # Sell Entry (ou Buy Entry invertido)
                mpf.make_addplot(bx_line, panel=1, color="r", linestyle=":"),  # Long Exit (ou Short Exit invertido)
                mpf.make_addplot(sx_line, panel=1, color="g", linestyle=":"),  # Short Exit (ou Long Exit invertido)
            ]
        )

        # Marcar pontos de entrada e saída do histórico
        for trade in trade_history:
            trade_time = trade["time"]
            trade_price = trade["price"]
            trade_type = trade["type"]

            # Encontra o índice (data) correspondente ao trade para plotar corretamente
            # Converte a hora do trade (que está em timezone local) para um datetime sem timezone (naive),
            # que é o formato do índice do nosso gráfico.
            trade_time_naive = trade_time.tz_convert("UTC").tz_localize(None)
            # Encontrar o índice mais próximo de forma compatível com versões antigas do pandas
            try:
                trade_idx = df_chart.index.get_indexer([trade_time_naive], method="nearest")[0]
                if trade_idx == -1:
                    raise ValueError("no nearest index found")
            except Exception:
                idx_values = df_chart.index.values.astype("datetime64[ns]")
                trade_idx = int(np.argmin(np.abs(idx_values - np.datetime64(trade_time_naive))))
            scatter_data = [np.nan] * len(df_chart)
            scatter_data[trade_idx] = trade_price

            # Marcadores no painel de preço
            if trade_type == "buy_entry":
                rsi_plots.append(mpf.make_addplot(scatter_data, type="scatter", marker="^", color="green", s=160))
            elif trade_type == "buy_exit":
                rsi_plots.append(mpf.make_addplot(scatter_data, type="scatter", marker="x", color="green", s=160))
            elif trade_type == "sell_entry":
                rsi_plots.append(mpf.make_addplot(scatter_data, type="scatter", marker="v", color="red", s=160))
            elif trade_type == "sell_exit":
                rsi_plots.append(mpf.make_addplot(scatter_data, type="scatter", marker="x", color="red", s=160))

            # Marcadores também no painel de RSI, usando RSI curto por padrão (fallback para médio/long)
            rsi_marker = pd.Series(np.nan, index=df_chart.index)
            rsi_val = df_chart["RSI_short"].iloc[trade_idx]
            if pd.isna(rsi_val):
                rsi_val = df_chart["RSI_med"].iloc[trade_idx]
            if pd.isna(rsi_val):
                rsi_val = df_chart["RSI_long"].iloc[trade_idx]
            if not pd.isna(rsi_val):
                rsi_marker.iloc[trade_idx] = rsi_val
                if trade_type == "buy_entry":
                    rsi_plots.append(
                        mpf.make_addplot(rsi_marker, panel=1, type="scatter", marker="^", color="green", s=120)
                    )
                elif trade_type == "buy_exit":
                    rsi_plots.append(
                        mpf.make_addplot(rsi_marker, panel=1, type="scatter", marker="x", color="green", s=120)
                    )
                elif trade_type == "sell_entry":
                    rsi_plots.append(
                        mpf.make_addplot(rsi_marker, panel=1, type="scatter", marker="v", color="red", s=120)
                    )
                elif trade_type == "sell_exit":
                    rsi_plots.append(
                        mpf.make_addplot(rsi_marker, panel=1, type="scatter", marker="x", color="red", s=120)
                    )

        # Pega a posição atual a partir do último trade no histórico
        current_pos = "NEUTRA"
        if trade_history:
            last_trade = trade_history[-1]["type"]
            if "entry" in last_trade:
                current_pos = "LONG" if "buy" in last_trade else "SHORT"

        # Título e Legendas
        title = f"{ticker} - {latest_time.strftime('%Y-%m-%d %H:%M:%S')} | Posição: {current_pos}"

        # Salvar o gráfico
        chart_dir = "live_charts"
        os.makedirs(chart_dir, exist_ok=True)
        filename = f"{chart_dir}/{ticker}_{latest_time.strftime('%Y%m%d_%H%M%S')}.png"

        # Plotar com retorno da figura para controlar eixos e salvar manualmente
        fig, axes = mpf.plot(
            df_chart,
            type="candle",
            style="yahoo",
            title=title,
            ylabel="Preço ($)",
            addplot=rsi_plots,
            # Dois painéis: preço (0) e RSI (1)
            panel_ratios=(3, 1),
            figscale=1.5,
            volume=False,
            warn_too_much_data=10000,
            returnfig=True,
        )

        # Ajusta eixo do painel de RSI para 0–100 e oculta o eixo esquerdo
        try:
            ax_rsi = axes[1] if isinstance(axes, (list, tuple)) else axes
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_yticks([0, 20, 40, 60, 80, 100])
            ax_rsi.tick_params(left=False, labelleft=False)
        except Exception:
            pass

        os.makedirs(chart_dir, exist_ok=True)
        fig.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico salvo em: {filename}")

    print("Iniciando modo live para Triple RSI.")
    print(f"Capital inicial: $ {initial_capital} | Invertido: {invert}")
    print("Status: Pressione Ctrl+C para parar.")
    print("-" * 50)

    position = 0  # 1 long, -1 short, 0 none
    entry_price = 0
    realized_pnl = 0
    trade_history = []

    while True:
        try:
            # Fetch latest data
            # Buscamos um período suficiente para calcular o RSI mais longo.
            # Se history_days for fornecido (ex.: vindo da configuração ativa), amplia a janela para reproduzir o estudo.
            min_hours = long_period * 4
            extra_hours = (history_days * 24) if history_days else 0
            hours_to_fetch = max(min_hours, extra_hours)
            hours_to_fetch = hours_to_fetch if hours_to_fetch > 0 else min_hours
            start_date = f"{hours_to_fetch} hours ago UTC"
            df = get_historical_klines(ticker, f"{interval_minutes}m", start_date)

            if df is None or len(df) < long_period + 1:
                print(
                    f"{datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S')}: Erro ao obter dados ou dados insuficientes. Tentando novamente..."
                )
                time.sleep(60)
                continue

            # Get latest values
            closes = df["close"]
            latest_close = closes.iloc[-1]
            # Converte o horário para o fuso local
            latest_time_utc = df["Date"].iloc[-1]
            latest_time_local = latest_time_utc.tz_localize("UTC").astimezone(local_tz)

            # Calcula todos os RSIs para o plot
            rsi_short_full = calculate_rsi(closes, short_period)
            rsi_med_full = calculate_rsi(closes, med_period)
            rsi_long_full = calculate_rsi(closes, long_period)

            # Últimos valores dos RSIs (reaproveitando os já calculados)
            rsi_short = rsi_short_full.iloc[-1]
            rsi_med = rsi_med_full.iloc[-1]
            rsi_long = rsi_long_full.iloc[-1]

            # Sinais usando helpers para reduzir duplicação e manter paridade com o backtest
            def all_below(x):
                return (rsi_short < x) and (rsi_med < x) and (rsi_long < x)

            def all_above(x):
                return (rsi_short > x) and (rsi_med > x) and (rsi_long > x)

            def any_below(x):
                return (rsi_short < x) or (rsi_med < x) or (rsi_long < x)

            def any_above(x):
                return (rsi_short > x) or (rsi_med > x) or (rsi_long > x)

            if invert:
                buy_signal = all_above(sell_entry)
                sell_signal = all_below(buy_entry)
                long_exit = any_below(sell_exit)
                short_exit = any_above(buy_exit)
            else:
                buy_signal = all_below(buy_entry)
                sell_signal = all_above(sell_entry)
                long_exit = any_above(buy_exit)
                short_exit = any_below(sell_exit)

            # Unrealized P&L
            if position == 1:
                unrealized_pnl = (latest_close - entry_price) * lot_size
            elif position == -1:
                unrealized_pnl = (entry_price - latest_close) * lot_size
            else:
                unrealized_pnl = 0
            current_capital = initial_capital + realized_pnl + unrealized_pnl

            # Entry
            if position == 0:
                if buy_signal:
                    position = 1
                    entry_price = latest_close
                    trade_history.append({"time": latest_time_local, "price": latest_close, "type": "buy_entry"})
                    print(
                        f"\n>>> {latest_time_local.strftime('%Y-%m-%d %H:%M:%S')}: SINAL COMPRA! Preço: $ {latest_close:.2f}, RSIs: {rsi_short:.2f}, {rsi_med:.2f}, {rsi_long:.2f}"
                    )
                elif sell_signal:
                    position = -1
                    entry_price = latest_close
                    trade_history.append({"time": latest_time_local, "price": latest_close, "type": "sell_entry"})
                    print(
                        f"\n>>> {latest_time_local.strftime('%Y-%m-%d %H:%M:%S')}: SINAL VENDA! Preço: $ {latest_close:.2f}, RSIs: {rsi_short:.2f}, {rsi_med:.2f}, {rsi_long:.2f}"
                    )

            # Exit
            elif position == 1:
                if long_exit:  # Use the pre-calculated boolean
                    realized_pnl += (latest_close - entry_price) * lot_size
                    trade_history.append({"time": latest_time_local, "price": latest_close, "type": "sell_exit"})
                    print(
                        f"\n<<< {latest_time_local.strftime('%Y-%m-%d %H:%M:%S')}: SAÍDA COMPRA! Preço: $ {latest_close:.2f}, P&L do Trade: $ {(latest_close - entry_price) * lot_size:.2f}"
                    )
                    position = 0
                    entry_price = 0
            elif position == -1:
                if short_exit:  # Use the pre-calculated boolean
                    realized_pnl += (entry_price - latest_close) * lot_size
                    trade_history.append({"time": latest_time_local, "price": latest_close, "type": "buy_exit"})
                    print(
                        f"\n<<< {latest_time_local.strftime('%Y-%m-%d %H:%M:%S')}: SAÍDA VENDA! Preço: $ {latest_close:.2f}, P&L do Trade: $ {(entry_price - latest_close) * lot_size:.2f}"
                    )
                    position = 0
                    entry_price = 0

            # Print status
            pos_string = "LONG" if position == 1 else "SHORT" if position == -1 else "NEUTRA"
            status = (
                f"{latest_time_local.strftime('%Y-%m-%d %H:%M:%S')} | Preço: ${latest_close:.2f} | Posição: {pos_string} | "
                f"Capital: ${current_capital:.2f} (P&L Realizado: ${realized_pnl:.2f}) | "
                f"RSIs(7,14,21): {rsi_short:.2f}, {rsi_med:.2f}, {rsi_long:.2f}"
            )
            print(status)  # Imprime o status atual

            # Gera e salva o gráfico do estado atual
            plot_live_status(df, rsi_short_full, rsi_med_full, rsi_long_full, trade_history, latest_time_local)

            # --- Lógica de Espera Sincronizada ---
            now = datetime.now(local_tz)  # Usamos o timezone local para calcular a próxima vela
            buffer_seconds = 15  # Margem para garantir que a nova vela esteja disponível

            # Calcula quantos minutos faltam para a próxima marca de 15 minutos
            minutes_to_wait = interval_minutes - (now.minute % interval_minutes)

            # Calcula o timestamp exato da próxima execução
            next_run_time = (now + timedelta(minutes=minutes_to_wait)).replace(second=0, microsecond=0) + timedelta(
                seconds=buffer_seconds
            )

            sleep_duration = (next_run_time - now).total_seconds()

            print(
                f"Verificação concluída às {now.strftime('%H:%M:%S')}. Próxima verificação aprox. às {next_run_time.strftime('%H:%M:%S')}..."
            )
            time.sleep(max(0, sleep_duration))  # Garante que o tempo de espera não seja negativo

        except KeyboardInterrupt:
            print("\nParando modo live.")
            print(f"Capital final: $ {initial_capital + realized_pnl:.2f}")
            break
        except Exception as e:
            print(f"Erro: {e}")
            time.sleep(60)  # Espera 1 minuto em caso de erro


if __name__ == "__main__":
    # Tenta carregar configuração ativa gerada pelo otimizador; fallback para defaults
    default_ticker = "BTCUSDT"
    default_interval = "5m"
    cfg = load_active_config(default_ticker, default_interval)
    if cfg:
        print(f"Usando configuração ativa de reports/active/{default_ticker}_{default_interval}.json")
        kwargs = cfg.to_live_kwargs()
        # Defaults de capital/lote se não vierem do relatório
        kwargs.setdefault("initial_capital", 1000)
        kwargs.setdefault("lot_size", 0.001)
        print(
            "Parâmetros carregados: short={s}, med={m}, long={l}, be={be}, se={se}, bx={bx}, sx={sx}, invert={inv}, days={days}".format(
                s=cfg.short_period,
                m=cfg.med_period,
                l=cfg.long_period,
                be=cfg.buy_entry,
                se=cfg.sell_entry,
                bx=cfg.buy_exit,
                sx=cfg.sell_exit,
                inv=cfg.invert,
                days=cfg.days if cfg.days else "-",
            )
        )
        live_triple_rsi(**kwargs, history_days=cfg.days)
    else:
        # Config padrão
        live_triple_rsi(
            ticker=default_ticker,
            initial_capital=1000,
            lot_size=0.001,
            invert=True,
            interval_minutes=5,
        )
