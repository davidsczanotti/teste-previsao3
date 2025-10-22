from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf


def plot_training_loss(losses: Sequence[float], out_path: str | None = None, return_fig: bool = False):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(losses, label="Train loss (MSE noise)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Diffusion Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
    if return_fig:
        return fig
    plt.close(fig)


def plot_predictions_on_candles(
    df: pd.DataFrame,
    last_lookback: int,
    future_dates: pd.DatetimeIndex,
    samples_close_paths: np.ndarray,
    out_path: str | None = None,
    title: str = "Diffusion Forecast (Close Paths)",
) -> plt.Figure | None:
    """
    - df: full dataframe, already sorted by Date
    - last_lookback: how many candles to show before forecast
    - future_dates: DatetimeIndex for H future steps
    - samples_close_paths: [N, H] array of future close levels (not returns)
    """
    df_tail = df.iloc[-last_lookback:].copy()
    df_tail = df_tail.set_index("Date")

    # Prepare candlestick plot
    style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle="--")
    fig, axlist = mpf.plot(
        df_tail[["open", "high", "low", "close", "volume"]],
        type="candle",
        volume=True,
        returnfig=True,
        style=style,
        title=title,
        figsize=(14, 8),
    )
    ax_price = axlist[0]

    # Overlay future samples (lines) on price axis
    N, H = samples_close_paths.shape
    for i in range(min(N, 30)):  # plot at most 30 to keep it readable
        ax_price.plot(future_dates, samples_close_paths[i], color="tab:blue", alpha=0.25, linewidth=1)

    # Fan chart (quantiles)
    qs = np.quantile(samples_close_paths, [0.1, 0.5, 0.9], axis=0)
    ax_price.fill_between(future_dates, qs[0], qs[2], color="tab:orange", alpha=0.2, label="P10-P90")
    ax_price.plot(future_dates, qs[1], color="tab:orange", linewidth=2, label="Median")

    ax_price.legend(loc="upper left")
    # Use subplots_adjust instead of tight_layout to avoid warnings with mpf figures
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.08, hspace=0.15)
    if out_path:
        fig.savefig(out_path)
    # Caller decides whether to close
    return fig


def plot_prob_next_return_positive(sampled_returns: np.ndarray, out_path: str) -> None:
    """
    sampled_returns: [N, H] array, we use step-1 for immediate probability.
    """
    step1 = sampled_returns[:, 0]
    p_up = np.mean(step1 > 0)
    plt.figure(figsize=(4, 4))
    plt.bar(["P(r1>0)"], [p_up], color="tab:green")
    plt.ylim(0, 1)
    plt.title("Probability Next Return > 0")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
