from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import List, Dict

import numpy as np
import pandas as pd
import pytz
import time
import os
import mplfinance as mpf

from ...binance_client import get_historical_klines
from .config import load_active_config


@dataclass
class StochParams:
    k_period: int = 9
    overbought: float = 75.0
    oversold: float = 25.0
    d_period: int = 3
    use_kd_cross: bool = True
    ema_period: int | None = None
    confirm_bars: int = 0
    cooldown_bars: int = 0
    min_hold_bars: int = 0


def stochastic_k(df: pd.DataFrame, k_period: int) -> pd.Series:
    close = df["close"].astype(float)
    ll = df["low"].rolling(window=k_period, min_periods=k_period).min()
    hh = df["high"].rolling(window=k_period, min_periods=k_period).max()
    rng = (hh - ll)
    k = pd.Series(np.where(rng.values == 0, 50.0, (close - ll) / rng * 100.0), index=close.index)
    return k


def cross_up(prev: float, cur: float, level: float) -> bool:
    return (prev < level) and (cur >= level)


def cross_down(prev: float, cur: float, level: float) -> bool:
    return (prev > level) and (cur <= level)


def plot_status(
    ticker: str,
    df: pd.DataFrame,
    k_series: pd.Series,
    params: StochParams,
    trade_history: List[Dict],
    latest_time_local: datetime,
):
    dfc = df.set_index("Date").rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    # Limitar quantidade para manter legibilidade
    if len(dfc) > 300:
        dfc = dfc.tail(300)

    addplots = [
        mpf.make_addplot(pd.Series(k_series.values[-len(dfc) :], index=dfc.index), panel=1, color="blue"),
        mpf.make_addplot(pd.Series(params.oversold, index=dfc.index), panel=1, color="green", linestyle="--"),
        mpf.make_addplot(pd.Series(params.overbought, index=dfc.index), panel=1, color="red", linestyle="--"),
    ]

    # Marcações de trades
    for tr in trade_history:
        t_local = tr["time"]
        px = tr["price"]
        kind = tr["type"]
        t_naive = t_local.tz_convert("UTC").tz_localize(None)
        # índice mais próximo
        try:
            idx = dfc.index.get_indexer([t_naive], method="nearest")[0]
            if idx == -1:
                raise ValueError
        except Exception:
            idx_values = dfc.index.values.astype("datetime64[ns]")
            idx = int(np.argmin(np.abs(idx_values - np.datetime64(t_naive))))
        markers = [np.nan] * len(dfc)
        markers[idx] = px
        if kind == "buy_entry":
            addplots.append(mpf.make_addplot(markers, type="scatter", marker="^", color="green", s=120))
        elif kind == "sell_entry":
            addplots.append(mpf.make_addplot(markers, type="scatter", marker="v", color="red", s=120))
        elif kind in ("buy_exit", "sell_exit"):
            addplots.append(mpf.make_addplot(markers, type="scatter", marker="x", color="black", s=120))

    title = f"{ticker} | {latest_time_local.strftime('%Y-%m-%d %H:%M:%S')}"
    fig, axes = mpf.plot(
        dfc,
        type="candle",
        style="yahoo",
        title=title,
        ylabel="Preço",
        addplot=addplots,
        panel_ratios=(3, 1),
        volume=False,
        returnfig=True,
    )

    try:
        ax_stoch = axes[1] if isinstance(axes, (list, tuple)) else axes
        ax_stoch.set_ylim(0, 100)
        ax_stoch.set_yticks([0, 20, 40, 60, 80, 100])
        ax_stoch.tick_params(left=False, labelleft=False)
    except Exception:
        pass

    os.makedirs("live_charts", exist_ok=True)
    out = f"live_charts/{ticker}_{latest_time_local.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    print(f"Gráfico salvo em: {out}")


def live_stochastic(
    ticker: str = "BTCUSDT",
    interval_minutes: int = 5,
    initial_capital: float = 1_000.0,
    lot_size: float = 0.001,
    timezone: str = "America/Sao_Paulo",
    params: StochParams = StochParams(),
    history_days: int | None = None,
):
    local_tz = pytz.timezone(timezone)

    print("Iniciando modo live — Estocástico (BTC/USDT, Binance)")
    print(f"Parâmetros: K={params.k_period}, OS={params.oversold}, OB={params.overbought}")
    print("Pressione Ctrl+C para parar.")

    position = 0
    entry_price = 0.0
    realized_pnl = 0.0
    trade_history: List[Dict] = []
    cooldown_until: datetime | None = None
    hold_bars = 0

    while True:
        try:
            # Janela mínima para calcular %K estável
            min_hours = params.k_period * 4
            extra_hours = (history_days * 24) if history_days else 0
            hours_to_fetch = max(min_hours, extra_hours)
            start_str = f"{hours_to_fetch} hours ago UTC"

            df = get_historical_klines(ticker, f"{interval_minutes}m", start_str)
            if df is None or df.empty or len(df) < params.k_period + 2:
                print(f"{datetime.now(local_tz).strftime('%H:%M:%S')}: Dados insuficientes. Retentando...")
                time.sleep(60)
                continue

            k_series = stochastic_k(df, params.k_period)
            d_series = k_series.rolling(window=params.d_period, min_periods=params.d_period).mean()
            ema_series = None
            if params.ema_period and params.ema_period > 0:
                ema_series = df["close"].ewm(span=params.ema_period, adjust=False).mean()

            # Confirmação por barras: usa deslocamento de 1 barra se confirm_bars=1
            off = params.confirm_bars
            if len(k_series) < off + 2:
                time.sleep(10)
                continue
            k_prev = float(k_series.iloc[-(off + 2)]) if not np.isnan(k_series.iloc[-(off + 2)]) else None
            k_cur = float(k_series.iloc[-(off + 1)]) if not np.isnan(k_series.iloc[-(off + 1)]) else None
            d_prev = float(d_series.iloc[-(off + 2)]) if not np.isnan(d_series.iloc[-(off + 2)]) else k_prev
            d_cur = float(d_series.iloc[-(off + 1)]) if not np.isnan(d_series.iloc[-(off + 1)]) else k_cur
            if (k_prev is None) or (k_cur is None):
                time.sleep(10)
                continue

            price = float(df["close"].iloc[-1])
            t_utc = df["Date"].iloc[-1]
            t_local = t_utc.tz_localize("UTC").astimezone(local_tz)

            # Sinais
            buy_entry = cross_up(k_prev, k_cur, params.oversold)
            sell_entry = cross_down(k_prev, k_cur, params.overbought)
            if params.use_kd_cross:
                kd_up = (k_prev - d_prev) <= 0 and (k_cur - d_cur) > 0
                kd_down = (k_prev - d_prev) >= 0 and (k_cur - d_cur) < 0
                buy_entry = buy_entry and kd_up
                sell_entry = sell_entry and kd_down
            long_exit_tp = cross_up(k_prev, k_cur, params.overbought)
            long_exit_stop = cross_down(k_prev, k_cur, params.oversold)
            short_exit_tp = cross_down(k_prev, k_cur, params.oversold)
            short_exit_stop = cross_up(k_prev, k_cur, params.overbought)

            # Filtro de tendência por EMA
            trend_ok_long = True
            trend_ok_short = True
            if ema_series is not None and not np.isnan(ema_series.iloc[-1]):
                ema_val = float(ema_series.iloc[-1])
                trend_ok_long = latest_close > ema_val
                trend_ok_short = latest_close < ema_val

            # Posição atual e PnL não realizado
            if position == 1:
                unreal = (price - entry_price) * lot_size
            elif position == -1:
                unreal = (entry_price - price) * lot_size
            else:
                unreal = 0.0
            capital_now = initial_capital + realized_pnl + unreal

            # Entradas
            # Cooldown de entradas
            if position == 0:
                if cooldown_until and latest_time_local < cooldown_until:
                    pass
                elif buy_entry and trend_ok_long:
                    position = 1
                    entry_price = price
                    trade_history.append({"time": t_local, "price": price, "type": "buy_entry"})
                    print(f">>> {t_local.strftime('%Y-%m-%d %H:%M:%S')}: COMPRA @ ${price:.2f} | %K: {k_prev:.2f}->{k_cur:.2f}")
                elif sell_entry and trend_ok_short:
                    position = -1
                    entry_price = price
                    trade_history.append({"time": t_local, "price": price, "type": "sell_entry"})
                    print(f">>> {t_local.strftime('%Y-%m-%d %H:%M:%S')}: VENDA @ ${price:.2f} | %K: {k_prev:.2f}->{k_cur:.2f}")

            # Saídas
            elif position == 1:
                if hold_bars >= params.min_hold_bars and (long_exit_tp or long_exit_stop):
                    realized = (price - entry_price) * lot_size
                    realized_pnl += realized
                    trade_history.append({"time": t_local, "price": price, "type": "sell_exit"})
                    print(
                        f"<<< {t_local.strftime('%Y-%m-%d %H:%M:%S')}: SAÍDA COMPRA @ ${price:.2f} | P&L: ${realized:.2f}"
                    )
                    position = 0
                    entry_price = 0.0
                    # inicia cooldown
                    cooldown_until = latest_time_local + timedelta(minutes=params.cooldown_bars * interval_minutes)
                    hold_bars = 0
            elif position == -1:
                if hold_bars >= params.min_hold_bars and (short_exit_tp or short_exit_stop):
                    realized = (entry_price - price) * lot_size
                    realized_pnl += realized
                    trade_history.append({"time": t_local, "price": price, "type": "buy_exit"})
                    print(
                        f"<<< {t_local.strftime('%Y-%m-%d %H:%M:%S')}: SAÍDA VENDA @ ${price:.2f} | P&L: ${realized:.2f}"
                    )
                    position = 0
                    entry_price = 0.0
                    cooldown_until = latest_time_local + timedelta(minutes=params.cooldown_bars * interval_minutes)
                    hold_bars = 0

            pos_str = "LONG" if position == 1 else ("SHORT" if position == -1 else "NEUTRA")
            print(
                f"{t_local.strftime('%Y-%m-%d %H:%M:%S')} | Preço: ${price:.2f} | Pos: {pos_str} | Capital: ${capital_now:.2f} | %K: {k_cur:.2f}"
            )

            # Plot do estado atual
            plot_status(ticker, df, k_series, params, trade_history, t_local)

            # Aguardar próximo candle com pequena margem
            now = datetime.now(local_tz)
            buffer = 15
            minutes_to_wait = interval_minutes - (now.minute % interval_minutes)
            next_run = (now + timedelta(minutes=minutes_to_wait)).replace(second=0, microsecond=0) + timedelta(
                seconds=buffer
            )
            sleep_s = max(0, (next_run - now).total_seconds())
            print(f"Próxima verificação ~ {next_run.strftime('%H:%M:%S')}")
            # incrementa hold quando em posição
            if position != 0:
                hold_bars += 1
            time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("\nParando modo live (Estocástico)")
            print(f"Capital final: ${initial_capital + realized_pnl:.2f}")
            break
        except Exception as e:
            print(f"Erro: {e}")
            time.sleep(60)


if __name__ == "__main__":
    # Tenta carregar configuração ativa do Estocástico; senão, usa defaults
    default_ticker = "BTCUSDT"
    default_interval = "5m"
    cfg = load_active_config(default_ticker, default_interval)
    if cfg:
        print(f"Usando configuração ativa STOCH: reports/active/STOCH_{cfg.ticker}_{cfg.interval}.json")
        live_stochastic(
            ticker=cfg.ticker,
            interval_minutes=cfg.interval_minutes,
            initial_capital=1_000.0,
            lot_size=cfg.lot_size if cfg.lot_size else 0.001,
            params=StochParams(
                k_period=cfg.k_period,
                oversold=cfg.oversold,
                overbought=cfg.overbought,
                d_period=cfg.d_period,
                use_kd_cross=cfg.use_kd_cross,
                ema_period=cfg.ema_period,
                confirm_bars=cfg.confirm_bars,
                cooldown_bars=cfg.cooldown_bars,
                min_hold_bars=cfg.min_hold_bars,
            ),
            history_days=cfg.days,
        )
    else:
        # Execução padrão: BTCUSDT 5m, estocástico 9, 25/75
        live_stochastic(
            ticker=default_ticker,
            interval_minutes=5,
            initial_capital=1_000.0,
            lot_size=0.001,
            params=StochParams(k_period=9, oversold=25.0, overbought=75.0, d_period=3, use_kd_cross=True),
        )
