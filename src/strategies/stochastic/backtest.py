from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from ...binance_client import get_historical_klines
from .config import load_active_config


@dataclass
class StochParams:
    k_period: int = 9
    overbought: float = 75.0
    oversold: float = 25.0
    d_period: int = 3  # suavização do %K para %D
    use_kd_cross: bool = True  # exige cruzamento K>D para compra e K<D para venda
    ema_period: int | None = None  # filtro de tendência
    confirm_bars: int = 0  # barras de confirmação após o cruzamento
    cooldown_bars: int = 0  # barras de espera após fechar um trade
    min_hold_bars: int = 0  # barras mínimas segurando a posição antes de permitir saída
    # ADX filter
    use_adx: bool = False
    adx_period: int = 14
    min_adx: float = 20.0


def stochastic_k(df: pd.DataFrame, k_period: int) -> pd.Series:
    """
    Calcula %K básico (sem média) usando período `k_period`.
    Fórmula: 100 * (Close - LL(k)) / (HH(k) - LL(k))
    Quando HH == LL, retorna 50 para evitar NaNs/inf.
    """
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


def backtest_stochastic(
    df: pd.DataFrame,
    params: StochParams = StochParams(),
    initial_capital: float = 10_000.0,
    lot_size: float = 0.1,
    fee_rate: float = 0.001,
) -> Tuple[List[Dict], float, Dict[str, float]]:
    """
    Backtest da estratégia do vídeo: entradas em cruzamentos das linhas 25/75.

    - Compra: %K cruza para cima a linha `oversold` (ex.: 25).
    - Saída da compra: %K cruza para cima `overbought` (TP) OU cruza para baixo `oversold` (stop).
    - Venda: %K cruza para baixo `overbought` (ex.: 75).
    - Saída da venda: %K cruza para baixo `oversold` (TP) OU cruza para cima `overbought` (stop).
    """
    df = df.sort_values("Date").reset_index(drop=True)
    k = stochastic_k(df, params.k_period)
    d = k.rolling(window=params.d_period, min_periods=params.d_period).mean()
    ema = None
    if params.ema_period and params.ema_period > 0:
        ema = df["close"].ewm(span=params.ema_period, adjust=False).mean()

    # ADX (opcional)
    adx = None
    if params.use_adx:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        period = max(2, params.adx_period)
        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        # Directional Movement
        plus_dm = (high.diff().clip(lower=0))
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0
        # Wilder's smoothing
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        pdi = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        mdi = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)

    # Índice inicial respeitando janelas
    start_idx = max(params.k_period, params.d_period, (params.ema_period or 0), params.confirm_bars + 1)

    # Desloca para obter valores anterior/atual para checar cruzamentos em fechamento com confirmação
    def series_at_offset(s: pd.Series, i: int, offset: int) -> float:
        return float(s.iloc[i - offset])

    position = 0  # 1 long, -1 short, 0 none
    entry_price = 0.0
    realized_pnl = 0.0
    equity_curve = [initial_capital]
    trades: List[Dict] = []
    cooldown_left = 0
    hold_bars = 0

    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(start_idx, len(df)):
        price = float(df["close"].iloc[i])
        t = df["Date"].iloc[i]

        kv_prev = series_at_offset(k, i, params.confirm_bars + 1)
        kv_cur = series_at_offset(k, i, params.confirm_bars)

        # Série D para cruzamento K/D
        dv_prev = series_at_offset(d, i, params.confirm_bars + 1) if not np.isnan(d.iloc[i]) else kv_prev
        dv_cur = series_at_offset(d, i, params.confirm_bars) if not np.isnan(d.iloc[i]) else kv_cur

        # Filtro de tendência por EMA (se disponível)
        trend_ok_long = True
        trend_ok_short = True
        if ema is not None and not np.isnan(ema.iloc[i]):
            ema_val = float(ema.iloc[i])
            trend_ok_long = price > ema_val
            trend_ok_short = price < ema_val
        if adx is not None and not np.isnan(adx.iloc[i]):
            if float(adx.iloc[i]) < params.min_adx:
                trend_ok_long = False
                trend_ok_short = False

        # Sinais base por cruzamentos com confirmação
        buy_entry = cross_up(kv_prev, kv_cur, params.oversold)
        sell_entry = cross_down(kv_prev, kv_cur, params.overbought)

        # Exigir cruzamento K/D na direção do sinal, se habilitado
        if params.use_kd_cross:
            kd_up = (kv_prev - dv_prev) <= 0 and (kv_cur - dv_cur) > 0
            kd_down = (kv_prev - dv_prev) >= 0 and (kv_cur - dv_cur) < 0
            buy_entry = buy_entry and kd_up
            sell_entry = sell_entry and kd_down

        long_exit_tp = cross_up(kv_prev, kv_cur, params.overbought)
        long_exit_stop = cross_down(kv_prev, kv_cur, params.oversold)
        short_exit_tp = cross_down(kv_prev, kv_cur, params.oversold)
        short_exit_stop = cross_up(kv_prev, kv_cur, params.overbought)

        # Atualiza cooldown/hold
        if cooldown_left > 0 and position == 0:
            cooldown_left -= 1

        if position != 0:
            hold_bars += 1

        if position == 0:
            if buy_entry:
                if cooldown_left == 0 and trend_ok_long:
                    position = 1
                    entry_price = price
                    hold_bars = 0
                    trades.append({"date": t, "action": "BUY", "price": price})
                    realized_pnl -= fee_rate * price * lot_size
            elif sell_entry:
                if cooldown_left == 0 and trend_ok_short:
                    position = -1
                    entry_price = price
                    hold_bars = 0
                    trades.append({"date": t, "action": "SELL", "price": price})
                    realized_pnl -= fee_rate * price * lot_size
        elif position == 1:
            can_exit = (hold_bars >= params.min_hold_bars)
            if can_exit and (long_exit_tp or long_exit_stop):
                pnl = (price - entry_price) * lot_size
                realized_pnl += pnl
                trades.append({"date": t, "action": "SELL", "price": price, "pnl": pnl})
                # taxa na saída
                realized_pnl -= fee_rate * price * lot_size
                if pnl >= 0:
                    gross_profit += pnl
                else:
                    gross_loss += -pnl
                position = 0
                entry_price = 0.0
                cooldown_left = params.cooldown_bars
                hold_bars = 0
        elif position == -1:
            can_exit = (hold_bars >= params.min_hold_bars)
            if can_exit and (short_exit_tp or short_exit_stop):
                pnl = (entry_price - price) * lot_size
                realized_pnl += pnl
                trades.append({"date": t, "action": "BUY_TO_COVER", "price": price, "pnl": pnl})
                realized_pnl -= fee_rate * price * lot_size
                if pnl >= 0:
                    gross_profit += pnl
                else:
                    gross_loss += -pnl
                position = 0
                entry_price = 0.0
                cooldown_left = params.cooldown_bars
                hold_bars = 0

        # Atualiza equity com PnL realizado + não realizado
        if position == 1:
            unreal = (price - entry_price) * lot_size
        elif position == -1:
            unreal = (entry_price - price) * lot_size
        else:
            unreal = 0.0
        equity_curve.append(initial_capital + realized_pnl + unreal)

    # Fecha posição remanescente ao final (no preço de fechamento final)
    if position != 0:
        last_price = float(df["close"].iloc[-1])
        t = df["Date"].iloc[-1]
        if position == 1:
            pnl = (last_price - entry_price) * lot_size
            realized_pnl += pnl
            trades.append({"date": t, "action": "SELL (final)", "price": last_price, "pnl": pnl})
            realized_pnl -= fee_rate * last_price * lot_size
            if pnl >= 0:
                gross_profit += pnl
            else:
                gross_loss += -pnl
        else:
            pnl = (entry_price - last_price) * lot_size
            realized_pnl += pnl
            trades.append({"date": t, "action": "BUY_TO_COVER (final)", "price": last_price, "pnl": pnl})
            realized_pnl -= fee_rate * last_price * lot_size
            if pnl >= 0:
                gross_profit += pnl
            else:
                gross_loss += -pnl
        equity_curve[-1] = initial_capital + realized_pnl

    # Métricas simples
    closed = [t for t in trades if "pnl" in t]
    n_trades = len(closed)
    wins = len([t for t in closed if t["pnl"] > 0])
    win_rate = (wins / n_trades * 100.0) if n_trades else 0.0
    total_pnl = float(realized_pnl)
    ret_pct = (total_pnl / initial_capital * 100.0) if initial_capital else 0.0

    # Drawdown (em %), Profit Factor e métricas adicionais
    running_max = np.maximum.accumulate(np.array(equity_curve, dtype=float))
    dd = (np.array(equity_curve) - running_max) / running_max * 100.0
    max_dd_pct = float(dd.min()) if len(dd) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    avg_pnl = (total_pnl / n_trades) if n_trades else 0.0

    stats = {
        "pnl": total_pnl,
        "num_trades": n_trades,
        "win_rate": win_rate,
        "return_pct": ret_pct,
        "profit_factor": profit_factor,
        "avg_pnl_per_trade": avg_pnl,
        "max_drawdown_pct": max_dd_pct,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "fee_rate": fee_rate,
    }

    # Plot opcional (desabilitado por padrão durante otimização)
    try:
        import os
        do_plot = bool(int(os.environ.get("STOCH_BACKTEST_PLOT", "1")))
    except Exception:
        do_plot = True
    if do_plot:
        try:
            import matplotlib.pyplot as plt

            fig, (ax_price, ax_stoch) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax_price.plot(df["Date"], df["close"], color="black", label="Close")
            ax_price.set_title("Stochastic Strategy — Price & Signals")
            ax_price.grid(True)

            ax_stoch.plot(df["Date"], k, label=f"%K ({params.k_period})", color="blue")
            ax_stoch.plot(df["Date"], d, label=f"%D ({params.d_period})", color="orange")
            ax_stoch.axhline(params.oversold, color="green", linestyle="--", label="Oversold")
            ax_stoch.axhline(params.overbought, color="red", linestyle="--", label="Overbought")
            ax_stoch.set_ylim(0, 100)
            ax_stoch.set_ylabel("%K")
            ax_stoch.grid(True)

            for tr in trades:
                if tr["action"] in ("BUY",):
                    ax_price.scatter(tr["date"], tr["price"], marker="^", color="green")
                elif tr["action"].startswith("SELL"):
                    ax_price.scatter(tr["date"], tr["price"], marker="v", color="red")
                elif tr["action"].startswith("BUY_TO_COVER"):
                    ax_price.scatter(tr["date"], tr["price"], marker="x", color="red")

            plt.tight_layout()
            plt.savefig("stochastic_backtest.png", dpi=120)
            plt.show()
        except Exception:
            pass

    # Prints resumidos
    print("Backtest Stochastic (K={k}, OB={ob}, OS={os}):".format(
        k=params.k_period, ob=params.overbought, os=params.oversold
    ))
    print("Total P&L: $ {0:.2f} ({1:.2f}%)".format(total_pnl, ret_pct))
    print("Trades fechados: {0} | Win rate: {1:.2f}%".format(n_trades, win_rate))
    print("PF: {0:.2f} | Avg/Trade: ${1:.4f} | MDD: {2:.2f}% | Fees: {3:.4%}".format(
        profit_factor if np.isfinite(profit_factor) else float('inf'), avg_pnl, max_dd_pct, fee_rate
    ))

    return trades, total_pnl, stats


if __name__ == "__main__":
    from datetime import datetime, timedelta, UTC

    default_ticker = "BTCUSDT"
    default_interval = "5m"
    cfg = load_active_config(default_ticker, default_interval)
    if cfg:
        days = cfg.days if cfg.days else 60
        start_dt = datetime.now(UTC) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Usando config ativa STOCH para {cfg.ticker}@{cfg.interval}. Baixando {days} dias de dados..."
        )
        df = get_historical_klines(cfg.ticker, cfg.interval, start_str)
        if df.empty:
            raise SystemExit("Falha ao obter dados da Binance.")
        params = StochParams(k_period=cfg.k_period, oversold=cfg.oversold, overbought=cfg.overbought)
        backtest_stochastic(
            df,
            params=params,
            initial_capital=1_000.0,
            lot_size=cfg.lot_size if cfg.lot_size else 0.001,
            fee_rate=cfg.fee_rate,
        )
    else:
        ticker = default_ticker
        interval = default_interval
        days = 60
        start_dt = datetime.now(UTC) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Buscando dados da Binance: {ticker} @ {interval} por {days} dias...")
        df = get_historical_klines(ticker, interval, start_str)
        if df.empty:
            raise SystemExit("Falha ao obter dados da Binance.")
        params = StochParams(k_period=9, overbought=75.0, oversold=25.0)
        backtest_stochastic(df, params=params, initial_capital=1_000.0, lot_size=0.001)
