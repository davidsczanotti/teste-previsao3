#!/usr/bin/env python3
"""
Visualização de backtest EMA-only com estilo próximo ao TradingView.

- Lê parâmetros de visualização de config.json (bloco "visualize").
- Usa mplfinance com tema escuro, candles e volume.
- Sobrepõe EMAs e, opcionalmente, EMA de referência e sinais.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .backtest import load_data_with_ref, calculate_mas, generate_signals


def _bars_per_day(timeframe: str) -> int:
    """Aproxima quantos candles existem por dia para um dado timeframe."""
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        minutes = int(tf[:-1])
        return max(1, 60 * 24 // minutes)
    if tf.endswith("h"):
        hours = int(tf[:-1])
        return max(1, 24 // hours)
    if tf.endswith("d"):
        return 1
    # fallback conservador
    return 24


def _build_style(viz_cfg: Dict[str, Any]):
    """
    Cria um estilo mplfinance inspirado no TradingView dark.
    Permite override futuro via config se necessário.
    """
    style_name = str(viz_cfg.get("style", "tradingview_dark")).lower()

    # Cores aproximadas do TradingView dark
    mc = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume="in",
    )

    if style_name in ("tradingview_dark", "tv_dark", "default"):
        return mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor="#131722",
            edgecolor="#131722",
            figcolor="#131722",
            gridcolor="#363c4e",
            gridstyle="-",
            y_on_right=True,
        )

    # Fallback para qualquer estilo não reconhecido
    return mpf.make_mpf_style(base_mpf_style=style_name, marketcolors=mc)


def plot_backtest(config_path: str = "src/strategies/ema_only/config.json") -> Path:
    """Plota gráfico do backtest com alta qualidade visual."""
    cfg_path = Path(config_path)
    with cfg_path.open() as f:
        config = json.load(f)

    viz_cfg: Dict[str, Any] = config.get("visualize", {})
    data_cfg: Dict[str, Any] = config.get("data", {})

    # Carregar dados + EMAs + sinais
    df = load_data_with_ref(config)
    df = calculate_mas(df, config)
    df = generate_signals(df, config)

    # Recorte de zoom por dias, respeitando timeframe
    zoom_days = viz_cfg.get("zoom_days", 60)
    if zoom_days and isinstance(zoom_days, (int, float)) and zoom_days > 0:
        bpd = _bars_per_day(str(data_cfg.get("timeframe", "1h")))
        df = df.tail(int(zoom_days * bpd))

    # Preparar dados para mplfinance
    if "Date" not in df.columns:
        raise ValueError("DataFrame não contém coluna 'Date' necessária para visualização.")

    df_plot = (
        df[["Date", "open", "high", "low", "close", "volume"]]
        .set_index("Date")
        .sort_index()
    )

    # Addplots: EMAs principais
    addplots: List[Any] = []
    if "ema_fast" in df.columns:
        addplots.append(mpf.make_addplot(df["ema_fast"], color="#4da6ff", width=1.0))
    if "ema_mid" in df.columns:
        addplots.append(mpf.make_addplot(df["ema_mid"], color="#ffd54f", width=1.0))
    if "ema_slow" in df.columns:
        addplots.append(mpf.make_addplot(df["ema_slow"], color="#ef5350", width=1.0))

    # EMA de referência (timeframe superior)
    if viz_cfg.get("show_ref_ema", True) and "ref_ema" in df.columns:
        addplots.append(mpf.make_addplot(df["ref_ema"], color="#9e9e9e", width=1.0, alpha=0.8))

    # Marcar sinais de entrada/saída
    if viz_cfg.get("show_signals", True) and "signal" in df.columns:
        longs = df["signal"] == 1
        shorts = df["signal"] == -1
        if longs.any():
            y_long = df["low"].copy().to_numpy(dtype=float)
            y_long[~longs.to_numpy()] = np.nan
            y_long = y_long * 0.995
            addplots.append(
                mpf.make_addplot(
                    y_long,
                    type="scatter",
                    markersize=40,
                    marker="^",
                    color="#6cff6c",
                )
            )
        if shorts.any():
            y_short = df["high"].copy().to_numpy(dtype=float)
            y_short[~shorts.to_numpy()] = np.nan
            y_short = y_short * 1.005
            addplots.append(
                mpf.make_addplot(
                    y_short,
                    type="scatter",
                    markersize=40,
                    marker="v",
                    color="#ff6666",
                )
            )

    style = _build_style(viz_cfg)

    # Tamanho da figura e DPI a partir do config
    fig_width = float(viz_cfg.get("figure_width", 16))
    fig_height = float(viz_cfg.get("figure_height", 9))
    dpi = int(viz_cfg.get("dpi", 160))

    # Diretório de saída (charts dedicado, alinhado com convenções do projeto)
    outdir = Path(viz_cfg.get("outdir", "src/strategies/ema_only/reports/charts"))
    outdir.mkdir(parents=True, exist_ok=True)

    symbol = str(data_cfg.get("symbol", "UNKNOWN"))
    timeframe = str(data_cfg.get("timeframe", "UNKNOWN"))
    output_file = outdir / f"ema_only_backtest_{symbol}_{timeframe}.png"

    fig, _ = mpf.plot(
        df_plot,
        type="candle",
        style=style,
        addplot=addplots if addplots else None,
        volume=bool(viz_cfg.get("show_volume", True)),
        figsize=(fig_width, fig_height),
        returnfig=True,
    )

    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfico salvo em {output_file}")
    return output_file


def main() -> None:
    config_path = Path(__file__).parent / "config.json"
    plot_backtest(str(config_path))


if __name__ == "__main__":
    main()
