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
from .webapp import LiveState, start_server
from torch.utils.tensorboard import SummaryWriter
import csv
from datetime import datetime, timezone


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
    serve_live: bool = False,
    server_host: str = "127.0.0.1",
    server_port: int = 5001,
    open_browser: bool = True,
    live_keep_snapshots: int = 12,
    target_mode: str = "close",
    web_history_candles: int = 500,
    ul_clip: float = 0.2,
    forecast_mode: str = "continuous",  # 'continuous' or 'cycle'
    cycle_steps: int | None = None,
    ret_clip_abs: float = 0.03,
    ret_clip_sigma: float = 3.0,
    log_tensorboard: bool = True,
    tb_logdir: str | Path = "runs/diffusion",
    tb_log_every: int = 50,
    tb_image_every: int = 250,
    log_csv: bool = True,
    csv_path: str | Path = "reports/diffusion_model/metrics_step.csv",
    audit_log_path: str | Path = "reports/diffusion_model/audit.ndjson",
    audit_save_dir: str | Path = "reports/diffusion_model/snapshots",
    audit_save_samples: int = 10,
    seed: int | None = 42,
    run_id: str | None = None,
):
    ensure_dir(out_dir)

    # Reproducibility and run id
    if seed is not None:
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
        except Exception:
            pass
    run_id_value = run_id or f"{symbol}_{timeframe}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    # Persist effective config
    try:
        cfg_out = Path(out_dir) / f"run_{run_id_value}_config.json"
        effective_cfg = {
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
            "visual_progress": visual_progress,
            "live_update_every": live_update_every,
            "live_num_samples": live_num_samples,
            "live_overlay_samples": live_overlay_samples,
            "live_overlay_alpha": live_overlay_alpha,
            "serve_live": serve_live,
            "server_host": server_host,
            "server_port": server_port,
            "target_mode": target_mode,
            "web_history_candles": web_history_candles,
            "ul_clip": ul_clip,
            "ret_clip_abs": ret_clip_abs,
            "ret_clip_sigma": ret_clip_sigma,
            "forecast_mode": forecast_mode,
            "cycle_steps": cycle_steps,
            "seed": seed,
            "run_id": run_id_value,
        }
        ensure_dir(cfg_out.parent)
        with open(cfg_out, "w") as f_cfg:
            json.dump(effective_cfg, f_cfg, indent=2)
    except Exception:
        pass

    # Data
    seq_cfg = SeqConfig(lookback=lookback, horizon=horizon, target_mode=target_mode)
    df, ds = prepare_data(symbol, timeframe, days, seq_cfg)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty. Increase 'days' or check cache.")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model
    device = make_device()
    cond_dim = ds.cond_dim
    d_y = 3 if target_mode == "ohlc3" else 1
    cond_encoder = CondEncoder(cond_dim, out_dim=128).to(device)
    eps_model = EpsModel(d_y=d_y, d_cond=128, model_dim=128, n_layers=2).to(device)
    params = list(cond_encoder.parameters()) + list(eps_model.parameters())
    optim = torch.optim.Adam(params, lr=lr)
    sched = DiffusionSchedule(DiffusionConfig(timesteps=diffusion_steps), device)

    # Return clipping threshold based on recent volatility
    try:
        closes_np = df["close"].astype(float).to_numpy()
        log_ret_hist = np.diff(np.log(closes_np + 1e-12))
        window = min(lookback, len(log_ret_hist))
        sigma_recent = np.nanstd(log_ret_hist[-window:]) if window > 0 else 0.0
        thr_sigma = float(ret_clip_sigma * sigma_recent) if sigma_recent > 0 else float(ret_clip_abs)
        ret_thr = float(min(ret_clip_abs, thr_sigma))
        if not np.isfinite(ret_thr) or ret_thr <= 0:
            ret_thr = float(ret_clip_abs)
    except Exception:
        ret_thr = float(ret_clip_abs)

    # Logging sinks
    writer: SummaryWriter | None = SummaryWriter(tb_logdir) if log_tensorboard else None
    csv_file = None
    csv_writer = None
    if log_csv:
        ensure_dir(Path(csv_path).parent)
        new_file = not Path(csv_path).exists()
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if new_file:
            csv_writer.writerow(["timestamp","epoch","step","global_step","loss"])
    ensure_dir(Path(audit_log_path).parent)
    ensure_dir(Path(audit_save_dir))

    def _audit_write(obj: dict):
        try:
            with open(audit_log_path, "a", encoding="utf-8") as f_audit:
                f_audit.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # Train
    losses: List[float] = []
    epoch_losses: List[List[float]] = []
    total_steps = 0
    # Keep recent forecast snapshots for persistence and coverage check
    forecast_snapshots: List[Dict[str, Any]] = []
    last_anchor_idx: int | None = None
    last_payload: Dict[str, Any] | None = None
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

    # Optional Flask server for browser-based live view
    live_state = LiveState()
    if serve_live:
        try:
            start_server(live_state, host=server_host, port=server_port, open_browser=open_browser)
            print(f"Live server em http://{server_host}:{server_port} (Poll /api/live)")
        except Exception as e:
            print(f"Falha ao iniciar o servidor Flask: {e}")
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
            # Log to TensorBoard/CSV
            if writer and (total_steps % tb_log_every == 0):
                writer.add_scalar("train/loss", float(loss), total_steps)
            if csv_writer and (total_steps % tb_log_every == 0):
                csv_writer.writerow([datetime.now(timezone.utc).isoformat(), epoch, i_batch, total_steps, float(loss)])
                csv_file.flush()
            # Live update every N steps (fast DDIM sampling, few samples)
            need_snapshot = (serve_live or live_view or (video_writer is not None))
            if need_snapshot and (total_steps % live_update_every == 0):
                with torch.no_grad():
                    anchor_idx = len(df) - 1
                    do_resample = True
                    if forecast_mode == "cycle":
                        min_steps = cycle_steps if cycle_steps is not None else ds.horizon
                        if last_anchor_idx is not None and (anchor_idx - last_anchor_idx) < int(min_steps):
                            do_resample = False

                    if not do_resample and serve_live and last_payload is not None:
                        # Repost last payload (keeps forecast stable) and skip to next batch
                        try:
                            live_state.set(last_payload)
                        except Exception:
                            pass
                        continue

                    # Resample a fresh forecast snapshot
                    cond_last_flat = torch.from_numpy(ds.xs[-1].reshape(-1)).unsqueeze(0).to(device)
                    cond_last_vec = cond_encoder(cond_last_flat)
                    y_samp = sample(
                        eps_model, sched, cond_last_vec.repeat(live_num_samples, 1), ds.horizon, d_y=d_y, ddim=True
                    ).cpu().numpy()  # [N,H,d_y]
                    candles_live = None
                    if d_y == 1:
                        y_ret_live = y_samp.squeeze(-1)
                        if ret_thr > 0:
                            y_ret_live = np.clip(y_ret_live, -ret_thr, ret_thr)
                        paths_live = float(df["close"].iloc[-1]) * np.exp(np.cumsum(y_ret_live, axis=1))
                    else:
                        dclose = y_samp[:, :, 0]
                        if ret_thr > 0:
                            dclose = np.clip(dclose, -ret_thr, ret_thr)
                        # relative log gaps with clamp to limit spikes
                        u_rel = np.minimum(np.log1p(np.exp(y_samp[:, :, 1])), ul_clip)
                        l_rel = np.minimum(np.log1p(np.exp(y_samp[:, :, 2])), ul_clip)
                        close_paths = float(df["close"].iloc[-1]) * np.exp(np.cumsum(dclose, axis=1))
                        opens = np.concatenate([
                            np.full((close_paths.shape[0], 1), float(df["close"].iloc[-1])),
                            close_paths[:, :-1]
                        ], axis=1)
                        highs = close_paths * np.exp(u_rel)
                        lows = close_paths * np.exp(-l_rel)
                        paths_live = close_paths
                        candles_live = {"open": opens, "high": highs, "low": lows, "close": close_paths}
                    last_date = pd.to_datetime(df["Date"].iloc[-1])
                    future_idx = ds.future_index(last_date, timeframe)
                    last_anchor_idx = anchor_idx
                    # Update on-screen matplotlib view if enabled
                    if live_view and (fig_live is not None):
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
                    # Update persistent snapshots list (median + band only)
                    try:
                        qs_med = np.median(paths_live, axis=0)
                        qs_p10 = np.quantile(paths_live, 0.1, axis=0)
                        qs_p90 = np.quantile(paths_live, 0.9, axis=0)
                        snap = {
                            "t0": pd.to_datetime(last_date),
                            "t": future_idx,
                            "median": qs_med,
                            "p10": qs_p10,
                            "p90": qs_p90,
                        }
                        forecast_snapshots.append(snap)
                        if len(forecast_snapshots) > live_keep_snapshots:
                            forecast_snapshots.pop(0)
                    except Exception:
                        pass

                    # Push snapshot to Flask live state (subset of overlays)
                    if serve_live:
                        try:
                            subset_n = min(paths_live.shape[0], live_overlay_samples)
                            # Compute coverage for the latest snapshot against realized closes
                            close_series = df.set_index("Date")["close"].astype(float)
                            hist_snaps_json = []
                            now = pd.to_datetime(df["Date"].iloc[-1])
                            for s in forecast_snapshots:
                                t = pd.DatetimeIndex(s["t"]).tz_localize(None)
                                # realized mask: future times that are now in history
                                mask = t <= now
                                n_obs = int(mask.sum())
                                hit_rate = None
                                if n_obs > 0:
                                    # align to exact timestamps; drop NaN due to any mismatch
                                    y_real = close_series.reindex(t[mask]).astype(float)
                                    ok = y_real.notna()
                                    n_obs = int(ok.sum())
                                    if n_obs > 0:
                                        lo = s["p10"][mask][ok.values]
                                        hi = s["p90"][mask][ok.values]
                                        yr = y_real[ok].values
                                        hit_rate = float(((yr >= lo) & (yr <= hi)).mean())
                                hist_snaps_json.append({
                                    "t0": pd.to_datetime(s["t0"]).isoformat(),
                                    "t": [pd.to_datetime(x).isoformat() for x in s["t"]],
                                    "median": s["median"].tolist(),
                                    "p10": s["p10"].tolist(),
                                    "p90": s["p90"].tolist(),
                                    "coverage": {"n_obs": n_obs, "hit_rate": hit_rate},
                                })
                            hw = int(web_history_candles)
                            hist_idx = df.index[-min(hw, len(df)) : ]
                            now_iso = pd.Timestamp.utcnow().isoformat()
                            # Explainability signals (trend, band width, expected return)
                            try:
                                log_close_hist = np.log(df.loc[hist_idx, "close"].astype(float).to_numpy() + 1e-12)
                                if len(log_close_hist) >= 2:
                                    slope = float((log_close_hist[-1] - log_close_hist[max(0, len(log_close_hist)-lookback)]) / max(1, min(lookback, len(log_close_hist)-1)))
                                else:
                                    slope = 0.0
                            except Exception:
                                slope = 0.0
                            band_width = float((qs_p90[-1] - qs_p10[-1]) / max(1e-9, qs_med[-1]))
                            exp_ret = float(qs_med[-1] / float(df["close"].iloc[-1]) - 1.0)
                            trend = "up" if slope > 0 else ("down" if slope < 0 else "flat")
                            explain = {
                                "trend": trend,
                                "slope": slope,
                                "exp_return": exp_ret,
                                "band_width": band_width,
                                "ret_clip_thr": ret_thr,
                                "ul_clip": ul_clip,
                            }
                            payload = {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "horizon": ds.horizon,
                                "target_mode": target_mode,
                                "loss": float(loss),
                                "ts": now_iso,
                                "history": {
                                    "t": [pd.to_datetime(x).isoformat() for x in df.loc[hist_idx, "Date"]],
                                    "open": df.loc[hist_idx, "open"].astype(float).tolist(),
                                    "high": df.loc[hist_idx, "high"].astype(float).tolist(),
                                    "low": df.loc[hist_idx, "low"].astype(float).tolist(),
                                    "close": df.loc[hist_idx, "close"].astype(float).tolist(),
                                },
                                "forecast": {
                                    "t": [pd.to_datetime(x).isoformat() for x in future_idx],
                                    "median": qs_med.tolist(),
                                    "p10": qs_p10.tolist(),
                                    "p90": qs_p90.tolist(),
                                    "paths": paths_live[:subset_n].tolist(),
                                    "history_snaps": hist_snaps_json,
                                    "candles": None,
                                },
                                "cycle": {
                                    "mode": forecast_mode,
                                    "anchor_idx": int(last_anchor_idx) if last_anchor_idx is not None else None,
                                    "required": int(cycle_steps if cycle_steps is not None else ds.horizon),
                                },
                                "explain": explain,
                            }
                            # Attach candles if available
                            try:
                                if candles_live is not None:
                                    o_med = np.median(candles_live["open"], axis=0).tolist()
                                    h_med = np.median(candles_live["high"], axis=0).tolist()
                                    l_med = np.median(candles_live["low"], axis=0).tolist()
                                    c_med = np.median(candles_live["close"], axis=0).tolist()
                                    o_s = candles_live["open"][:subset_n].tolist()
                                    h_s = candles_live["high"][:subset_n].tolist()
                                    l_s = candles_live["low"][:subset_n].tolist()
                                    c_s = candles_live["close"][:subset_n].tolist()
                                    payload["forecast"]["candles"] = {
                                        "median": {"open": o_med, "high": h_med, "low": l_med, "close": c_med},
                                        "samples": {"open": o_s, "high": h_s, "low": l_s, "close": c_s},
                                    }
                            except Exception:
                                pass
                            # Audit snapshot meta + save small arrays
                            try:
                                meta = {
                                    "type": "snapshot",
                                    "ts": now_iso,
                                    "run_id": _run_id,
                                    "horizon": int(ds.horizon),
                                    "num_samples": int(live_num_samples),
                                    "ret_clip_thr": float(ret_thr),
                                    "ul_clip": float(ul_clip),
                                    "median_last": float(qs_med[-1]),
                                    "p10_last": float(qs_p10[-1]),
                                    "p90_last": float(qs_p90[-1]),
                                    "explain": explain,
                                }
                                _audit_write(meta)
                                snap_path = Path(audit_save_dir) / f"snap_{_run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
                                save = {
                                    "t": [pd.to_datetime(x).isoformat() for x in future_idx],
                                    "median": qs_med.tolist(),
                                    "p10": qs_p10.tolist(),
                                    "p90": qs_p90.tolist(),
                                    "paths": paths_live[: min(audit_save_samples, paths_live.shape[0])].tolist(),
                                }
                                with open(snap_path, "w") as f:
                                    _json.dump(save, f)
                                live_state.set(payload, meta=meta)
                            except Exception:
                                live_state.set(payload)
                            last_payload = payload
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
                y_s = sample(
                    eps_model,
                    sched,
                    cond_last_vec.repeat(num_samples, 1),
                    ds.horizon,
                    d_y=d_y,
                    ddim=True,
                ).cpu().numpy()
                if d_y == 1:
                    y_returns = y_s.squeeze(-1)
                    if ret_thr > 0:
                        y_returns = np.clip(y_returns, -ret_thr, ret_thr)
                    paths = float(df["close"].iloc[-1]) * np.exp(np.cumsum(y_returns, axis=1))
                else:
                    dclose = y_s[:, :, 0]
                    if ret_thr > 0:
                        dclose = np.clip(dclose, -ret_thr, ret_thr)
                    paths = float(df["close"].iloc[-1]) * np.exp(np.cumsum(dclose, axis=1))
                future_idx = ds.future_index(pd.to_datetime(df["Date"].iloc[-1]), timeframe)
                plot_predictions_on_candles(
                    df,
                    last_lookback=lookback,
                    future_dates=future_idx,
                    samples_close_paths=paths,
                    out_path=os.path.join(out_dir, f"forecast_progress_{symbol}_{timeframe}.png"),
                )
                if d_y == 1:
                    plot_prob_next_return_positive(
                        y_returns,
                        os.path.join(out_dir, f"prob_up_progress_{symbol}_{timeframe}.png"),
                    )
        # TensorBoard figures (at interval)
        if writer and (total_steps % tb_image_every == 0):
            try:
                fig1 = plot_training_loss(losses, out_path=None, return_fig=True)
                writer.add_figure("charts/loss", fig1, total_steps)
                plt.close(fig1)
                # Build quick forecast fig
                cond_last_flat = torch.from_numpy(ds.xs[-1].reshape(-1)).unsqueeze(0).to(device)
                cond_last_vec = cond_encoder(cond_last_flat)
                y_tmp = sample(eps_model, sched, cond_last_vec.repeat(min(32, num_samples), 1), ds.horizon, d_y=d_y, ddim=True)
                y_tmp = y_tmp.squeeze(-1).cpu().numpy() if d_y==1 else y_tmp[:, :, 0].cpu().numpy()
                if ret_thr>0:
                    y_tmp = np.clip(y_tmp, -ret_thr, ret_thr)
                paths_tmp = float(df["close"].iloc[-1]) * np.exp(np.cumsum(y_tmp, axis=1))
                future_idx = ds.future_index(pd.to_datetime(df["Date"].iloc[-1]), timeframe)
                fig2 = plot_predictions_on_candles(df, lookback, future_idx, paths_tmp, out_path=None)
                writer.add_figure("charts/forecast", fig2, total_steps)
                plt.close(fig2)
            except Exception:
                pass

    # Save training loss plot
    if save_final_images:
        plot_training_loss(losses, os.path.join(out_dir, f"training_loss_{symbol}_{timeframe}.png"))

    # Inference: take last window to condition and sample scenarios (final)
    last_idx = len(df) - 1
    # Build last conditioning vector from dataset arrays
    # Use the same transforms as dataset: last L window from normalized X already inside dataset.xs
    cond_last_flat = torch.from_numpy(ds.xs[-1].reshape(-1)).unsqueeze(0).to(device)
    cond_last_vec = cond_encoder(cond_last_flat)
    y_samp_final = sample(eps_model, sched, cond_last_vec.repeat(num_samples, 1), ds.horizon, d_y=d_y, ddim=True).cpu().numpy()
    # Convert to price paths (close only or candles)
    last_close = float(df["close"].iloc[-1])
    if d_y == 1:
        y_returns = y_samp_final.squeeze(-1)  # [N,H]
        if ret_thr > 0:
            y_returns = np.clip(y_returns, -ret_thr, ret_thr)
        factors = np.exp(np.cumsum(y_returns, axis=1))
        paths = last_close * factors
    else:
        dclose = y_samp_final[:, :, 0]
        if ret_thr > 0:
            dclose = np.clip(dclose, -ret_thr, ret_thr)
        u_rel = np.minimum(np.log1p(np.exp(y_samp_final[:, :, 1])), ul_clip)
        l_rel = np.minimum(np.log1p(np.exp(y_samp_final[:, :, 2])), ul_clip)
        close_paths = last_close * np.exp(np.cumsum(dclose, axis=1))
        paths = close_paths
        opens = np.concatenate([np.full((close_paths.shape[0], 1), last_close), close_paths[:, :-1]], axis=1)
        highs = close_paths * np.exp(u_rel)
        lows = close_paths * np.exp(-l_rel)

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
        if d_y == 1:
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
    if writer:
        try:
            writer.flush(); writer.close()
        except Exception:
            pass
    if csv_file:
        csv_file.close()


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
            "save_epoch_images": False,
            "save_final_images": True,
            "live_record_video": False,
            "live_video_path": "reports/diffusion_model/training_live.mp4",
            "live_fps": 5,
            "dark_mode": True,
            "live_overlay_samples": 20,
            "live_overlay_alpha": 0.2,
            "live_keep_snapshots": 12,
            "serve_live": True,
            "server_host": "127.0.0.1",
            "server_port": 5001,
            "open_browser": True,
            "target_mode": "close",
            "web_history_candles": 500,
            "ul_clip": 0.2,
            "forecast_mode": "cycle",
            "cycle_steps": 10,
            "ret_clip_abs": 0.03,
            "ret_clip_sigma": 3.0,
            "log_tensorboard": True,
            "tb_logdir": "runs/diffusion",
            "tb_log_every": 50,
            "tb_image_every": 250,
            "log_csv": True,
            "csv_path": "reports/diffusion_model/metrics_step.csv",
            "audit_log_path": "reports/diffusion_model/audit.ndjson",
            "audit_save_dir": "reports/diffusion_model/snapshots",
            "audit_save_samples": 10,
            "seed": 42,
            "run_id": None
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
        serve_live=bool(cfg.get("serve_live", True)),
        server_host=cfg.get("server_host", "127.0.0.1"),
        server_port=int(cfg.get("server_port", 5001)),
        open_browser=bool(cfg.get("open_browser", True)),
        target_mode=cfg.get("target_mode", "close"),
        web_history_candles=int(cfg.get("web_history_candles", 500)),
        ul_clip=float(cfg.get("ul_clip", 0.2)),
        forecast_mode=str(cfg.get("forecast_mode", "continuous")),
        cycle_steps=int(cfg.get("cycle_steps", 0)) or None,
        ret_clip_abs=float(cfg.get("ret_clip_abs", 0.03)),
        ret_clip_sigma=float(cfg.get("ret_clip_sigma", 3.0)),
        log_tensorboard=bool(cfg.get("log_tensorboard", True)),
        tb_logdir=cfg.get("tb_logdir", "runs/diffusion"),
        tb_log_every=int(cfg.get("tb_log_every", 50)),
        tb_image_every=int(cfg.get("tb_image_every", 250)),
        log_csv=bool(cfg.get("log_csv", True)),
        csv_path=cfg.get("csv_path", "reports/diffusion_model/metrics_step.csv"),
        audit_log_path=cfg.get("audit_log_path", "reports/diffusion_model/audit.ndjson"),
        audit_save_dir=cfg.get("audit_save_dir", "reports/diffusion_model/snapshots"),
        audit_save_samples=int(cfg.get("audit_save_samples", 10)),
        seed=cfg.get("seed", 42),
        run_id=cfg.get("run_id", None),
    )
