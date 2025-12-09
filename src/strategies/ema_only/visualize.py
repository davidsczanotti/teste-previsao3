from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import backtest_ema_only, compute_ema, compute_sma
from .optimize import prepare_dataset_with_reference, _build_params_from_config


def _split_trades_for_plot(
    trades: List[Dict[str, Any]],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Tuple[List[pd.Timestamp], List[float]]]:
    long_x: List[pd.Timestamp] = []
    long_y: List[float] = []
    short_x: List[pd.Timestamp] = []
    short_y: List[float] = []
    long_exit_x: List[pd.Timestamp] = []
    long_exit_y: List[float] = []
    short_exit_x: List[pd.Timestamp] = []
    short_exit_y: List[float] = []

    for t in trades:
        date = pd.to_datetime(t["date"])
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue
        price = float(t["price"])
        reason = str(t.get("reason", ""))
        action = str(t.get("action", "")).upper()

        # Entradas: motivos de sinal (signal_*) ou simples (simple_long/simple_short).
        is_signal_entry = reason.startswith("signal") or reason in {"simple_long", "simple_short"}

        if is_signal_entry:
            if action == "BUY":
                long_x.append(date)
                long_y.append(price)
            elif action == "SELL":
                short_x.append(date)
                short_y.append(price)
        else:
            # Saídas long: qualquer SELL que não seja sinal (stop/exit/etc).
            if action == "SELL":
                long_exit_x.append(date)
                long_exit_y.append(price)
            # Saídas short: qualquer BUY que não seja sinal (stop/exit/etc).
            elif action == "BUY":
                short_exit_x.append(date)
                short_exit_y.append(price)

    return {
        "long": (long_x, long_y),
        "short": (short_x, short_y),
        "long_exit": (long_exit_x, long_exit_y),
        "short_exit": (short_exit_x, short_exit_y),
    }


def main() -> None:
    """
    Gera um gráfico simples: close + EMAs + entradas/saídas da estratégia.

    Uso recomendado:
        BINANCE_OFFLINE=1 poetry run python -m src.strategies.ema_only.visualize
    """
    cfg_path = Path(__file__).with_name("config.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    data_cfg: Dict[str, Any] = cfg.get("data", {})
    strat_cfg: Dict[str, Any] = cfg.get("strategy", {})
    backtest_cfg: Dict[str, Any] = cfg.get("backtest", {})

    symbol = data_cfg["symbol"]
    timeframe = data_cfg["timeframe"]
    days = int(data_cfg.get("days", 365))
    ref_timeframe = data_cfg.get("ref_timeframe")
    ref_days = data_cfg.get("ref_days")
    ref_ema_period = strat_cfg.get("ref_ema_period")

    df = prepare_dataset_with_reference(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        use_cache_only=True,
        ref_timeframe=ref_timeframe,
        ref_days=ref_days,
        ref_ema_period=ref_ema_period,
    )

    params = _build_params_from_config(strat_cfg)
    initial_capital = float(backtest_cfg.get("initial_capital", 1_000.0))
    monthly_target_pct = float(backtest_cfg.get("monthly_target_pct", 0.01))

    trades, _, _ = backtest_ema_only(
        df,
        params=params,
        initial_capital=initial_capital,
        monthly_target_pct=monthly_target_pct,
    )

    # Séries de preço e EMAs principais.
    close = df["close"].astype(float)
    ema_fast = compute_ema(close, params.ema_period)
    ema_slow = compute_ema(close, params.slow_ema_period)
    ema_mid = compute_ema(close, params.ema_mid_period) if params.ema_mid_period else None
    sma_fast = compute_sma(close, params.sma_fast_period) if params.sma_fast_period else None
    sma_mid = compute_sma(close, params.sma_mid_period) if params.sma_mid_period else None
    sma_slow = compute_sma(close, params.sma_slow_period) if params.sma_slow_period else None
    ref_ema = df["ref_ema"] if "ref_ema" in df.columns else None

    # Marcadores de trade permanecem calculados, mas não são plotados nesta versão (foco em MAs).
    markers_full = _split_trades_for_plot(trades)

    # Estilos: EMAs tracejadas/pontilhadas, SMAs contínuas.
    ema_style = {"linestyle": "--", "alpha": 0.9, "linewidth": 1.5}
    sma_style = {"linestyle": "-", "alpha": 0.9, "linewidth": 1.5}

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], close, label="close", color="#4c78a8", linewidth=1.2)
    ax.plot(df["Date"], ema_fast, label="ema_fast", color="#f58518", **ema_style)
    if ema_mid is not None:
        ax.plot(df["Date"], ema_mid, label="ema_mid", color="#72b7b2", **ema_style)
    ax.plot(df["Date"], ema_slow, label="ema_slow", color="#e45756", **ema_style)
    if sma_fast is not None:
        ax.plot(df["Date"], sma_fast, label="sma_fast", color="#9c755f", **sma_style)
    if sma_mid is not None:
        ax.plot(df["Date"], sma_mid, label="sma_mid", color="#b279a2", **sma_style)
    if sma_slow is not None:
        ax.plot(df["Date"], sma_slow, label="sma_slow", color="#ff9da6", **sma_style)
    if ref_ema is not None:
        ax.plot(df["Date"], ref_ema, label="ref_ema", color="#666666", linestyle="--", linewidth=1.5, alpha=0.8)

    # Marcas de entrada/saída.
    long_x, long_y = markers_full["long"]
    short_x, short_y = markers_full["short"]
    long_exit_x, long_exit_y = markers_full["long_exit"]
    short_exit_x, short_exit_y = markers_full["short_exit"]

    if long_x:
        ax.scatter(long_x, long_y, marker="^", color="green", label="long")
    if short_x:
        ax.scatter(short_x, short_y, marker="v", color="red", label="short")
    if long_exit_x:
        ax.scatter(long_exit_x, long_exit_y, marker="x", color="green", label="long_exit")
    if short_exit_x:
        ax.scatter(short_exit_x, short_exit_y, marker="x", color="red", label="short_exit")

    ax.set_title(f"Ações da estratégia vs preço/EMAs — {symbol} {timeframe}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço")
    ax.legend()
    ax.grid(True, alpha=0.2)

    charts_dir = Path("src/strategies/ema_only/reports/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    out_path = charts_dir / f"ema_only_backtest_{symbol}_{timeframe}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    # Zoom em janela recente (ex.: últimos N dias) para enxergar melhor cruzamentos.
    viz_cfg: Dict[str, Any] = cfg.get("visualize", {})
    zoom_days = int(viz_cfg.get("zoom_days", 60))
    df_zoom: Optional[pd.DataFrame] = None
    if zoom_days > 0:
        end_ts = df["Date"].iloc[-1]
        start_ts = end_ts - pd.Timedelta(days=zoom_days)
        df_zoom = df[df["Date"] >= start_ts].copy()
        if not df_zoom.empty:
            close_z = df_zoom["close"].astype(float)
            ema_fast_z = ema_fast[df_zoom.index]
            ema_slow_z = ema_slow[df_zoom.index]
            ema_mid_z = ema_mid[df_zoom.index] if ema_mid is not None else None
            sma_fast_z = sma_fast[df_zoom.index] if sma_fast is not None else None
            sma_mid_z = sma_mid[df_zoom.index] if sma_mid is not None else None
            sma_slow_z = sma_slow[df_zoom.index] if sma_slow is not None else None
            ref_ema_z = ref_ema[df_zoom.index] if ref_ema is not None else None

            markers_zoom = _split_trades_for_plot(trades, start=start_ts, end=end_ts)

            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(df_zoom["Date"], close_z, label="close", color="#4c78a8", linewidth=1.2)
            ax2.plot(df_zoom["Date"], ema_fast_z, label="ema_fast", color="#f58518", **ema_style)
            if ema_mid_z is not None:
                ax2.plot(df_zoom["Date"], ema_mid_z, label="ema_mid", color="#72b7b2", **ema_style)
            ax2.plot(df_zoom["Date"], ema_slow_z, label="ema_slow", color="#e45756", **ema_style)
            if sma_fast_z is not None:
                ax2.plot(df_zoom["Date"], sma_fast_z, label="sma_fast", color="#9c755f", **sma_style)
            if sma_mid_z is not None:
                ax2.plot(df_zoom["Date"], sma_mid_z, label="sma_mid", color="#b279a2", **sma_style)
            if sma_slow_z is not None:
                ax2.plot(df_zoom["Date"], sma_slow_z, label="sma_slow", color="#ff9da6", **sma_style)
            if ref_ema_z is not None:
                ax2.plot(df_zoom["Date"], ref_ema_z, label="ref_ema", color="#666666", linestyle="--", linewidth=1.5, alpha=0.8)

            zx_long, zy_long = markers_zoom["long"]
            zx_short, zy_short = markers_zoom["short"]
            zx_long_exit, zy_long_exit = markers_zoom["long_exit"]
            zx_short_exit, zy_short_exit = markers_zoom["short_exit"]

            if zx_long:
                ax2.scatter(zx_long, zy_long, marker="^", color="green", label="long")
            if zx_short:
                ax2.scatter(zx_short, zy_short, marker="v", color="red", label="short")
            if zx_long_exit:
                ax2.scatter(zx_long_exit, zy_long_exit, marker="x", color="green", label="long_exit")
            if zx_short_exit:
                ax2.scatter(zx_short_exit, zy_short_exit, marker="x", color="red", label="short_exit")

            ax2.set_title(f"Ações vs 6 MAs (zoom {zoom_days}d) — {symbol} {timeframe}")
            ax2.set_xlabel("Data")
            ax2.set_ylabel("Preço")
            ax2.legend()
            ax2.grid(True, alpha=0.2)

            zoom_path = charts_dir / f"ema_only_backtest_{symbol}_{timeframe}_zoom.png"
            fig2.tight_layout()
            fig2.savefig(zoom_path)
            plt.close(fig2)

            print(f"[ema_only.visualize] Gráfico zoom salvo em: {zoom_path}")

    # Gráfico limpo: apenas close + MAs (sem marcadores), usando a janela de zoom se disponível.
    df_clean = df_zoom if df_zoom is not None and not df_zoom.empty else df
    clean_idx = df_clean.index
    close_c = close[clean_idx]
    ema_fast_c = ema_fast[clean_idx]
    ema_slow_c = ema_slow[clean_idx]
    ema_mid_c = ema_mid[clean_idx] if ema_mid is not None else None
    sma_fast_c = sma_fast[clean_idx] if sma_fast is not None else None
    sma_mid_c = sma_mid[clean_idx] if sma_mid is not None else None
    sma_slow_c = sma_slow[clean_idx] if sma_slow is not None else None
    ref_ema_c = ref_ema[clean_idx] if ref_ema is not None else None

    fig_clean, ax_clean = plt.subplots(figsize=(12, 5))
    ax_clean.plot(df_clean["Date"], close_c, label="close", color="#4c78a8", linewidth=1.2)
    ax_clean.plot(df_clean["Date"], ema_fast_c, label="ema_fast", color="#f58518", **ema_style)
    if ema_mid_c is not None:
        ax_clean.plot(df_clean["Date"], ema_mid_c, label="ema_mid", color="#72b7b2", **ema_style)
    ax_clean.plot(df_clean["Date"], ema_slow_c, label="ema_slow", color="#e45756", **ema_style)
    if sma_fast_c is not None:
        ax_clean.plot(df_clean["Date"], sma_fast_c, label="sma_fast", color="#9c755f", **sma_style)
    if sma_mid_c is not None:
        ax_clean.plot(df_clean["Date"], sma_mid_c, label="sma_mid", color="#b279a2", **sma_style)
    if sma_slow_c is not None:
        ax_clean.plot(df_clean["Date"], sma_slow_c, label="sma_slow", color="#ff9da6", **sma_style)
    if ref_ema_c is not None:
        ax_clean.plot(df_clean["Date"], ref_ema_c, label="ref_ema", color="#666666", linestyle="--", linewidth=1.5, alpha=0.8)

    ax_clean.set_title(f"Preço e 6 MAs (limpo zoom) — {symbol} {timeframe}")
    ax_clean.set_xlabel("Data")
    ax_clean.set_ylabel("Preço")
    ax_clean.legend()
    ax_clean.grid(True, alpha=0.2)

    clean_path = charts_dir / f"ema_only_backtest_{symbol}_{timeframe}_clean.png"
    fig_clean.tight_layout()
    fig_clean.savefig(clean_path)
    plt.close(fig_clean)

    print(f"[ema_only.visualize] Gráfico salvo em: {out_path}")
    print(f"[ema_only.visualize] Gráfico limpo salvo em: {clean_path}")


if __name__ == "__main__":
    main()
