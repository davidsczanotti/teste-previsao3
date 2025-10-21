from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ...utils.data_loader import load_data
from .dataset import OHLCVDiffusionDataset, SeqConfig
from .model import EpsModel, CondEncoder
from .diffusion import DiffusionConfig, DiffusionSchedule, training_step, sample
from .visualize import plot_training_loss, plot_predictions_on_candles, plot_prob_next_return_positive
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
from matplotlib.animation import writers as _anim_writers  # type: ignore


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(symbol: str, timeframe: str, days: int, cfg: SeqConfig) -> Tuple[pd.DataFrame, OHLCVDiffusionDataset]:
    df = load_data(symbol, timeframe, days=days, use_cache_only=True)
    df.attrs["ticker"] = symbol
    ds = OHLCVDiffusionDataset(df, cfg)
    ds.timeframe = timeframe
    return df, ds


def _setup_live_plot(
    df: pd.DataFrame,
    lookback: int,
    title: str,
    *,
    dark_mode: bool = True,
    show_volume: bool = False,
    overlay_n: int = 20,
    overlay_alpha: float = 0.2,
):
    """Create a live candlestick plot with multiple translucent forecast overlays.

    Returns fig, ax_price, lines_overlay(list), line_median, hist_left, band_handle(None)
    """
    plt.ion()
    df_tail = df.iloc[-lookback:].copy().set_index("Date")
    if dark_mode:
        style = mpf.make_mpf_style(base_mpf_style="nightclouds", gridstyle="-", gridcolor="0.2")
        figcolor = 'k'
        facecolor = 'k'
    else:
        style = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle="--")
        figcolor = 'w'
        facecolor = 'w'

    fig, axlist = mpf.plot(
        df_tail[["open", "high", "low", "close", "volume"]],
        type="candle",
        volume=show_volume,
        returnfig=True,
        style=style,
        title=title,
        figsize=(14, 8),
        figcolor=figcolor,
        facecolor=facecolor,
    )
    ax_price = axlist[0]
    # Clean, minimal look (like your reference)
    try:
        ax_price.grid(False)
    except Exception:
        pass
    # Create multiple translucent overlay lines for sampled paths
    lines_overlay: List[Any] = []
    for _ in range(overlay_n):
        (line,) = ax_price.plot_date([], [], '-', color=(0.8, 0.8, 0.8), alpha=overlay_alpha, linewidth=1)
        lines_overlay.append(line)
    # Median (highlight)
    (line_median,) = ax_price.plot_date([], [], '-', color="#00d1ff", alpha=0.9, linewidth=2, label="Forecast median")
    ax_price.legend(loc="upper left")
    fig.canvas.draw_idle()
    plt.show(block=False)
    hist_left = df_tail.index[0]
    band = None
    return fig, ax_price, lines_overlay, line_median, hist_left, band


def _update_live_projection(
    ax_price,
    lines_overlay: List[Any],
    line_median,
    hist_left: pd.Timestamp,
    future_idx: pd.DatetimeIndex,
    paths: np.ndarray,
    band,
):
    """Update overlays: multiple paths (translucent), median, and quantile band."""
    x = mdates.date2num(future_idx.to_pydatetime())
    N, H = paths.shape
    # Update overlay lines
    for i, line in enumerate(lines_overlay):
        if i < N:
            line.set_data(x, paths[i])
        else:
            line.set_data([], [])
    # Median + band
    qs = np.quantile(paths, [0.1, 0.5, 0.9], axis=0)
    line_median.set_data(x, qs[1])
    if band is not None:
        try:
            band.remove()
        except Exception:
            pass
        band = None
    band = ax_price.fill_between(x, qs[0], qs[2], color="#00d1ff", alpha=0.10, linewidth=0)
    # Axes
    ax_price.set_xlim(mdates.date2num(hist_left.to_pydatetime()), x[-1])
    ax_price.relim()
    ax_price.autoscale_view(scalex=False, scaley=True)
    ax_price.figure.canvas.draw_idle()
    plt.pause(0.001)
    return band


def train_and_visualize(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    days: int = 120,
    lookback: int = 64,
    horizon: int = 16,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    diffusion_steps: int = 50,
    out_dir: str | Path = "reports/diffusion_model",
    num_samples: int = 64,
    visual_progress: bool = True,
    live_update_every: int = 50,
    live_num_samples: int = 16,
    save_epoch_images: bool = False,
    save_final_images: bool = True,
    live_record_video: bool = False,
    live_video_path: str | Path = "reports/diffusion_model/training_live.mp4",
    live_fps: int = 5,
    dark_mode: bool = True,
    live_overlay_samples: int = 20,
    live_overlay_alpha: float = 0.2,
):
    ensure_dir(out_dir)

    # Data
    seq_cfg = SeqConfig(lookback=lookback, horizon=horizon)
    df, ds = prepare_data(symbol, timeframe, days, seq_cfg)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty. Increase 'days' or check cache.")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model
    device = make_device()
    cond_dim = ds.cond_dim
    d_y = 1
    cond_encoder = CondEncoder(cond_dim, out_dim=128).to(device)
    eps_model = EpsModel(d_y=d_y, d_cond=128, model_dim=128, n_layers=2).to(device)
    params = list(cond_encoder.parameters()) + list(eps_model.parameters())
    optim = torch.optim.Adam(params, lr=lr)
    sched = DiffusionSchedule(DiffusionConfig(timesteps=diffusion_steps), device)

    # Train
    losses: List[float] = []
    epoch_losses: List[List[float]] = []
    total_steps = 0
    # Live view setup (controlled via config/args) + backend detection
    live_view = bool(visual_progress)
    backend = str(mpl.get_backend()).lower()
    non_gui = any(k in backend for k in ["agg", "pdf", "svg", "ps", "cairo", "template"])  # common non-GUI backends
    if live_view and non_gui:
        print(f"Matplotlib backend '{mpl.get_backend()}' não suporta janela interativa; habilitando gravação do vídeo do live.")
        live_view = False
        # If no GUI, at least record a single video/gif if not requested
        if not live_record_video:
            live_record_video = True
    # Create plot if we either want to view live or record a video
    fig_live = ax_price_live = lines_live = line_live_med = hist_left = band_live = None
    try:
        if live_view or live_record_video:
            fig_live, ax_price_live, lines_live, line_live_med, hist_left, band_live = _setup_live_plot(
                df,
                lookback,
                title=f"{symbol} {timeframe} — Live Forecast",
                dark_mode=dark_mode,
                show_volume=False,
                overlay_n=live_overlay_samples,
                overlay_alpha=live_overlay_alpha,
            )
    except Exception as e:
        print(f"Falha ao iniciar o live plot: {e}")
        live_view = False
        live_record_video = False
    # Optionally open a video writer (single file) if requested
    video_writer = None
    video_is_gif = False
    if (live_record_video and fig_live is not None):
        ensure_dir(Path(live_video_path).parent)
        # Prefer FFMPEG if available, else try Pillow (GIF)
        try:
            if 'ffmpeg' in _anim_writers.list():
                FFMpegWriter = _anim_writers['ffmpeg']
                video_writer = FFMpegWriter(fps=live_fps)
                video_writer.setup(fig_live, str(live_video_path), dpi=100)
            else:
                raise RuntimeError('ffmpeg writer not available')
        except Exception:
            try:
                if 'pillow' in _anim_writers.list():
                    PillowWriter = _anim_writers['pillow']
                    gif_path = str(Path(out_dir) / 'training_live.gif')
                    live_video_path = gif_path
                    video_writer = PillowWriter(fps=live_fps)
                    video_writer.setup(fig_live, gif_path, dpi=100)
                    video_is_gif = True
                else:
                    print('Live recording disabled: no video writer (ffmpeg/pillow) available.')
            except Exception:
                print('Live recording disabled: error initializing video writer.')

    for epoch in range(1, epochs + 1):
        epoch_loss_vals: List[float] = []
        for i_batch, (cond_flat, y) in enumerate(dl, start=1):
            cond_flat = cond_flat.to(device)
            y = y.to(device)
            cond_vec = cond_encoder(cond_flat)
            loss = training_step(eps_model, sched, optim, y, cond_vec)
            losses.append(loss)
            epoch_loss_vals.append(loss)
            total_steps += 1
            if total_steps % 50 == 0:
                print(f"epoch {epoch} step {total_steps} loss {loss:.6f}")
            # Live update every N steps (fast DDIM sampling, few samples)
            if live_view and (total_steps % live_update_every == 0):
                with torch.no_grad():
                    cond_last_flat = torch.from_numpy(ds.xs[-1].reshape(-1)).unsqueeze(0).to(device)
                    cond_last_vec = cond_encoder(cond_last_flat)
                    y_ret_live = sample(
                        eps_model, sched, cond_last_vec.repeat(live_num_samples, 1), ds.horizon, d_y=d_y, ddim=True
                    ).squeeze(-1).cpu().numpy()
                    paths_live = float(df["close"].iloc[-1]) * np.exp(np.cumsum(y_ret_live, axis=1))
                    last_date = pd.to_datetime(df["Date"].iloc[-1])
                    future_idx = ds.future_index(last_date, timeframe)
                    band_live = _update_live_projection(
                        ax_price_live,
                        lines_live,
                        line_live_med,
                        hist_left,
                        future_idx,
                        paths_live,
                        band_live,
                    )
                    # Record a frame into a single video file if enabled
                    if video_writer is not None:
                        try:
                            video_writer.grab_frame()
                        except Exception:
                            pass
        epoch_losses.append(epoch_loss_vals)

        # Save intermediate visuals per época — opcional
        if visual_progress and save_epoch_images:
            # Loss até aqui
            plot_training_loss(losses, os.path.join(out_dir, f"training_progress_{symbol}_{timeframe}.png"))
            # Snapshot simples (uma imagem que é sobrescrita):
            with torch.no_grad():
                cond_last_flat = torch.from_numpy(ds.xs[-1].reshape(-1)).unsqueeze(0).to(device)
                cond_last_vec = cond_encoder(cond_last_flat)
                y_returns = sample(
                    eps_model,
                    sched,
                    cond_last_vec.repeat(num_samples, 1),
                    ds.horizon,
                    d_y=d_y,
                    ddim=True,
                )
                y_returns = y_returns.squeeze(-1).cpu().numpy()  # [N,H]
                paths = float(df["close"].iloc[-1]) * np.exp(np.cumsum(y_returns, axis=1))
                future_idx = ds.future_index(pd.to_datetime(df["Date"].iloc[-1]), timeframe)
                plot_predictions_on_candles(
                    df,
                    last_lookback=lookback,
                    future_dates=future_idx,
                    samples_close_paths=paths,
                    out_path=os.path.join(out_dir, f"forecast_progress_{symbol}_{timeframe}.png"),
                )
                plot_prob_next_return_positive(
                    y_returns,
                    os.path.join(out_dir, f"prob_up_progress_{symbol}_{timeframe}.png"),
                )

    # Save training loss plot
    if save_final_images:
        plot_training_loss(losses, os.path.join(out_dir, f"training_loss_{symbol}_{timeframe}.png"))

    # Inference: take last window to condition and sample scenarios (final)
    last_idx = len(df) - 1
    # Build last conditioning vector from dataset arrays
    # Use the same transforms as dataset: last L window from normalized X already inside dataset.xs
    cond_last_flat = torch.from_numpy(ds.xs[-1].reshape(-1)).unsqueeze(0).to(device)
    cond_last_vec = cond_encoder(cond_last_flat)
    y_returns = sample(eps_model, sched, cond_last_vec.repeat(num_samples, 1), ds.horizon, d_y=d_y, ddim=True)
    y_returns = y_returns.squeeze(-1).cpu().numpy()  # [N,H]

    # Convert returns to price paths
    last_close = float(df["close"].iloc[-1])
    # cumulative sum of log returns => multiplicative factor
    factors = np.exp(np.cumsum(y_returns, axis=1))
    paths = last_close * factors  # [N,H]

    # Future dates
    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_idx = ds.future_index(last_date, timeframe)

    # Plots
    if save_final_images:
        plot_predictions_on_candles(
            df,
            last_lookback=lookback,
            future_dates=future_idx,
            samples_close_paths=paths,
            out_path=os.path.join(out_dir, f"forecast_paths_{symbol}_{timeframe}.png"),
        )
        plot_prob_next_return_positive(y_returns, os.path.join(out_dir, f"prob_up_{symbol}_{timeframe}.png"))

    # Save metrics metadata to JSON
    metrics = {
        "symbol": symbol,
        "timeframe": timeframe,
        "days": days,
        "lookback": lookback,
        "horizon": horizon,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "diffusion_steps": diffusion_steps,
        "num_samples": num_samples,
        "total_steps": total_steps,
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
        "epoch_losses": epoch_losses,
    }
    with open(os.path.join(out_dir, f"metrics_{symbol}_{timeframe}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved figures and metrics to: {Path(out_dir).resolve()}")
    # Keep live window open briefly after finishing
    try:
        if live_view:
            print("Training completed. Keeping live window for 5 seconds...")
            plt.pause(5)
    except Exception:
        pass
    # Finalize video recording if active
    if video_writer is not None:
        try:
            video_writer.finish()
            print(f"Saved live training video to: {live_video_path}")
        except Exception:
            pass


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.json")


def load_config(path: Path | None = None) -> Dict[str, Any]:
    cfg_path = path or _default_config_path()
    if not cfg_path.exists():
        # Fallback defaults if config is missing
        return {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "days": 180,
            "lookback": 64,
            "horizon": 16,
            "epochs": 3,
            "batch_size": 64,
            "lr": 1e-3,
            "diffusion_steps": 50,
            "out_dir": "reports/diffusion_model",
            "num_samples": 64,
            "visual_progress": True,
            "live_update_every": 50,
            "live_num_samples": 16,
            "save_epoch_images": false,
            "save_final_images": true,
            "live_record_video": false,
            "live_video_path": "reports/diffusion_model/training_live.mp4",
            "live_fps": 5,
            "dark_mode": true,
            "live_overlay_samples": 20,
            "live_overlay_alpha": 0.2
        }
    with open(cfg_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # Load config from JSON (no CLI flags)
    cfg = load_config()
    train_and_visualize(
        symbol=cfg.get("symbol", "BTCUSDT"),
        timeframe=cfg.get("timeframe", "1h"),
        days=int(cfg.get("days", 180)),
        lookback=int(cfg.get("lookback", 64)),
        horizon=int(cfg.get("horizon", 16)),
        epochs=int(cfg.get("epochs", 3)),
        batch_size=int(cfg.get("batch_size", 64)),
        lr=float(cfg.get("lr", 1e-3)),
        diffusion_steps=int(cfg.get("diffusion_steps", 50)),
        out_dir=cfg.get("out_dir", "reports/diffusion_model"),
        num_samples=int(cfg.get("num_samples", 64)),
        visual_progress=bool(cfg.get("visual_progress", True)),
        live_update_every=int(cfg.get("live_update_every", 50)),
        live_num_samples=int(cfg.get("live_num_samples", 16)),
        save_epoch_images=bool(cfg.get("save_epoch_images", False)),
        save_final_images=bool(cfg.get("save_final_images", True)),
        live_record_video=bool(cfg.get("live_record_video", False)),
        live_video_path=cfg.get("live_video_path", "reports/diffusion_model/training_live.mp4"),
        live_fps=int(cfg.get("live_fps", 5)),
        dark_mode=bool(cfg.get("dark_mode", True)),
        live_overlay_samples=int(cfg.get("live_overlay_samples", 20)),
        live_overlay_alpha=float(cfg.get("live_overlay_alpha", 0.2)),
    )
