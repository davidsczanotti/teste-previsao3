from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

from .data import load_primary_series, load_confirm_series, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy, PPOConfig
from .utils_cfg import build_policy
from .trainer import PPOTrainer
from ...utils.metrics import calculate_metrics


@dataclass
class WFWindow:
    train_start: int
    train_end: int
    val_start: int
    val_end: int


def build_windows(length: int, train_len: int, val_len: int, step: int) -> List[WFWindow]:
    windows: List[WFWindow] = []
    start = 0
    while start + train_len + val_len < length:
        windows.append(
            WFWindow(
                train_start=start,
                train_end=start + train_len,
                val_start=start + train_len,
                val_end=start + train_len + val_len,
            )
        )
        start += step
    return windows


DEFAULT_CONFIG = Path("src/strategies/exper_corr_pos/config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for MoE PPO agent")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to JSON configuration (default: src/strategies/exper_corr_pos/config.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    wf_cfg = cfg.get("walk_forward", {})
    env_cfg = EnvConfig(**cfg.get("env", {}))
    model_cfg = cfg.get("model", {})
    ppo_cfg = PPOConfig(**cfg.get("ppo", {}))

    primary_df = load_primary_series(cfg)
    confirm_df = load_confirm_series(cfg)
    dataset = prepare_dataset(primary_df, config=cfg, confirm_df=confirm_df)
    price_cols = ["open", "high", "low", "close", "volume"]
    timestamps = dataset.index.to_list()
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    train_hours = int(wf_cfg.get("train_days", 720)) * 24
    val_hours = int(wf_cfg.get("val_days", 180)) * 24
    step_hours = int(wf_cfg.get("step_days", 90)) * 24

    windows = build_windows(len(price_df), train_hours, val_hours, step_hours)
    outdir = Path(wf_cfg.get("outdir", "src/strategies/exper_corr_pos/reports/walk_forward"))
    outdir.mkdir(parents=True, exist_ok=True)

    progress_path = outdir / "wf_progress.log"
    progress_path.write_text("")

    def _log(msg: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        line = f"{timestamp} {msg}"
        print(line, flush=True)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = torch.device(wf_cfg.get("device", "cpu"))
    histories = []
    total_windows = len(windows)

    _log(
        f"[WF] Iniciando walk-forward com {total_windows} janelas "
        f"(episodes={wf_cfg.get('episodes', 200)}, rollout_steps={wf_cfg.get('rollout_steps', 2048)})"
    )

    # Avaliador de um ambiente (executa política greedy e coleta métricas por trades)
    def _eval_env(policy: MoEPolicy, env: BTCMixtureEnv) -> Dict[str, Any]:
        obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
        rewards = []
        done = False
        trades: List[Dict[str, Any]] = []
        while not done:
            dist, _, _ = policy(obs.unsqueeze(0))
            action = torch.argmax(dist.probs, dim=-1).item()
            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            if info.get("trade_closed"):
                trades.append(
                    {
                        "pnl": float(info.get("trade_pnl", 0.0)),
                        "duration_bars": int(info.get("trade_bars", 0)),
                    }
                )
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        metrics = calculate_metrics(trades)
        durs = [t.get("duration_bars", 0) for t in trades if t.get("duration_bars", 0) > 0]
        avg_dur_bars = float(np.mean(durs)) if durs else 0.0
        avg_dur_hours = avg_dur_bars  # timeframe 1h
        return {
            "pnl": float(np.sum(rewards)),
            "equity_end": float(info.get("equity", 0.0)),
            "trades": int(len(trades)),
            "trade_list": trades,
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0)),
            "total_pnl_trades": float(metrics.get("total_pnl", 0.0)),
            "avg_win": float(metrics.get("avg_win", 0.0)),
            "avg_loss": float(metrics.get("avg_loss", 0.0)),
            "avg_trade_duration_bars": avg_dur_bars,
            "avg_trade_duration_hours": avg_dur_hours,
        }

    for idx, window in enumerate(windows, start=1):
        _log(
            f"[WF] Janela {idx}/{total_windows} — treino [{window.train_start}:{window.train_end}) "
            f"val [{window.val_start}:{window.val_end})"
        )
        train_prices = price_df.iloc[window.train_start : window.train_end]
        train_feats = feat_df.iloc[window.train_start : window.train_end]
        val_prices = price_df.iloc[window.val_start : window.val_end]
        val_feats = feat_df.iloc[window.val_start : window.val_end]

        # Ajusta normalização com base APENAS no treino deste window
        train_norm_mean = train_feats.mean()
        train_norm_std = train_feats.std().replace(0.0, 1.0)

        train_ts = timestamps[window.train_start : window.train_end]
        env = BTCMixtureEnv(
            train_prices,
            train_feats,
            env_cfg,
            norm_mean=train_norm_mean,
            norm_std=train_norm_std,
            timestamps=train_ts,
        )
        input_dim = train_feats.shape[1]
        policy = build_policy(input_dim, cfg)
        trainer = PPOTrainer(policy, ppo_cfg, device=device)

        episodes = int(wf_cfg.get("episodes", 200))
        rollout_steps = int(wf_cfg.get("rollout_steps", 2048))
        for episode in range(episodes):
            trainer.train_step(env, rollout_steps)
            if (episode + 1) % 20 == 0 or episode + 1 == episodes:
                _log(f"[WF] Janela {idx}/{total_windows} — episódio {episode + 1}/{episodes}")

        # Evaluate on validation (sem vazamento): usa normalização do treino
        val_ts = timestamps[window.val_start : window.val_end]
        val_env_train_norm = BTCMixtureEnv(
            val_prices,
            val_feats,
            env_cfg,
            norm_mean=train_norm_mean,
            norm_std=train_norm_std,
            timestamps=val_ts,
        )
        res_train_norm = _eval_env(policy, val_env_train_norm)

        # Evaluate on validation com normalização por janelão de validação (pode vazar info)
        val_env_val_norm = BTCMixtureEnv(val_prices, val_feats, env_cfg, timestamps=val_ts)
        res_val_norm = _eval_env(policy, val_env_val_norm)

        # Lag test: usa features defasadas em 1 barra para ver sensibilidade a timing
        lag_feats = val_feats.shift(1).dropna()
        if not lag_feats.empty:
            lag_prices = val_prices.iloc[val_feats.shape[0] - lag_feats.shape[0] :]
            lag_ts = val_ts[len(val_ts) - len(lag_feats) :]
            val_env_lag = BTCMixtureEnv(
                lag_prices,
                lag_feats,
                env_cfg,
                norm_mean=train_norm_mean,
                norm_std=train_norm_std,
                timestamps=lag_ts,
            )
            res_lag = _eval_env(policy, val_env_lag)
        else:
            res_lag = {"pnl": 0.0, "equity_end": 0.0, "trades": 0}

        # Heurísticas simples de alerta (não-fatais):
        norm_delta = res_val_norm["equity_end"] - res_train_norm["equity_end"]
        norm_leak_suspected = norm_delta > 0.05 * max(env_cfg.init_equity, 1e-6)
        lag_drop = res_train_norm["equity_end"] - res_lag["equity_end"]
        timing_suspected = lag_drop > 0.1 * max(env_cfg.init_equity, 1e-6)

        histories.append(
            {
                "window": idx,
                "train_norm": res_train_norm,
                "val_norm": res_val_norm,
                "lag1": res_lag,
                "norm_delta_equity": norm_delta,
                "norm_leak_suspected": bool(norm_leak_suspected),
                "lag_equity_drop": lag_drop,
                "timing_suspected": bool(timing_suspected),
            }
        )

        ckpt = outdir / f"policy_window{idx}.pt"
        torch.save(policy.state_dict(), ckpt)
        _log(
            "[WF] Janela "
            f"{idx}/{total_windows} concluída — eq_train={res_train_norm['equity_end']:.2f} "
            f"eq_val={res_val_norm['equity_end']:.2f} trades_val={res_train_norm['trades']}"
        )

    # Estatísticas agregadas no estilo do al_brooks
    val_metrics = [h["val_norm"] for h in histories]
    total_periods = len(val_metrics)
    periods_with_profit = sum(1 for m in val_metrics if m.get("pnl", 0.0) > 0)
    periods_with_loss = sum(1 for m in val_metrics if m.get("pnl", 0.0) <= 0)
    success_rate = (periods_with_profit / total_periods) if total_periods > 0 else 0.0

    total_pnl = float(sum(m.get("pnl", 0.0) for m in val_metrics))
    avg_pnl = float(total_pnl / total_periods) if total_periods > 0 else 0.0
    avg_win_rate = float(np.mean([m.get("win_rate", 0.0) for m in val_metrics])) if total_periods > 0 else 0.0
    avg_profit_factor = float(
        np.mean([m.get("profit_factor", 0.0) for m in val_metrics])
    ) if total_periods > 0 else 0.0
    total_trades = int(sum(m.get("trades", 0) for m in val_metrics))
    avg_trade_duration_bars = float(
        np.mean([m.get("avg_trade_duration_bars", 0.0) for m in val_metrics])
    ) if total_periods > 0 else 0.0
    avg_trade_duration_hours = float(
        np.mean([m.get("avg_trade_duration_hours", 0.0) for m in val_metrics])
    ) if total_periods > 0 else 0.0

    summary = {
        "total_periods": total_periods,
        "successful_periods": periods_with_profit,
        "success_rate": success_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_win_rate": avg_win_rate,
        "avg_profit_factor": avg_profit_factor,
        "total_trades": total_trades,
        "avg_trade_duration_bars": avg_trade_duration_bars,
        "avg_trade_duration_hours": avg_trade_duration_hours,
        "periods_with_profit": periods_with_profit,
        "periods_with_loss": periods_with_loss,
    }

    payload = {"summary": summary, "windows": histories}
    summary_path = outdir / "wf_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    _log(
        f"[WF] Concluído. Períodos={total_periods} trades={total_trades} "
        f"P&L_total={total_pnl:.2f} win_rate_médio={avg_win_rate:.2%} PF_médio={avg_profit_factor:.2f}"
    )
    _log(f"[WF] Resumo salvo em {summary_path}")


if __name__ == "__main__":
    main()
