from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import torch

from .data import load_btc_1h, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy, PPOConfig
from .trainer import PPOTrainer


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


DEFAULT_CONFIG = Path("src/strategies/exper_corr_neg/config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for MoE PPO agent")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to JSON configuration (default: src/strategies/exper_corr_neg/config.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    wf_cfg = cfg.get("walk_forward", {})
    env_cfg = EnvConfig(**cfg.get("env", {}))
    model_cfg = cfg.get("model", {})
    ppo_cfg = PPOConfig(**cfg.get("ppo", {}))

    df = load_btc_1h(days=3650)
    dataset = prepare_dataset(df)
    price_cols = ["open", "high", "low", "close", "volume"]
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    train_hours = int(wf_cfg.get("train_days", 720)) * 24
    val_hours = int(wf_cfg.get("val_days", 180)) * 24
    step_hours = int(wf_cfg.get("step_days", 90)) * 24

    windows = build_windows(len(price_df), train_hours, val_hours, step_hours)
    outdir = Path(wf_cfg.get("outdir", "reports/exper_corr_neg/walk_forward"))
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

        env = BTCMixtureEnv(
            train_prices,
            train_feats,
            env_cfg,
            norm_mean=train_norm_mean,
            norm_std=train_norm_std,
        )
        input_dim = train_feats.shape[1]
        policy = MoEPolicy(
            input_dim=input_dim,
            num_actions=3,
            expert_hidden=model_cfg.get("expert_hidden", [64, 32]),
            gating_hidden=model_cfg.get("gating_hidden", [64, 32]),
            num_experts=model_cfg.get("num_experts", 5),
            temperature=model_cfg.get("temperature", 0.7),
            top_k=model_cfg.get("top_k", 2),
        )
        trainer = PPOTrainer(policy, ppo_cfg, device=device)

        episodes = int(wf_cfg.get("episodes", 200))
        rollout_steps = int(wf_cfg.get("rollout_steps", 2048))
        for episode in range(episodes):
            trainer.train_step(env, rollout_steps)
            if (episode + 1) % 20 == 0 or episode + 1 == episodes:
                _log(f"[WF] Janela {idx}/{total_windows} — episódio {episode + 1}/{episodes}")

        # Evaluate on validation (sem vazamento): usa normalização do treino
        def _eval_env(policy: MoEPolicy, env: BTCMixtureEnv) -> dict:
            obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
            rewards = []
            done = False
            trades = 0
            prev_pos = 0
            while not done:
                dist, _, _ = policy(obs.unsqueeze(0))
                action = torch.argmax(dist.probs, dim=-1).item()
                desired_pos = action - 1
                if desired_pos != prev_pos and desired_pos != 0:
                    trades += 1
                next_obs, reward, done, info = env.step(action)
                rewards.append(reward)
                obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
                prev_pos = desired_pos
            return {
                "pnl": float(np.sum(rewards)),
                "equity_end": float(info.get("equity", 0.0)),
                "trades": int(trades),
            }

        val_env_train_norm = BTCMixtureEnv(
            val_prices,
            val_feats,
            env_cfg,
            norm_mean=train_norm_mean,
            norm_std=train_norm_std,
        )
        res_train_norm = _eval_env(policy, val_env_train_norm)

        # Evaluate on validation com normalização por janelão de validação (pode vazar info)
        val_env_val_norm = BTCMixtureEnv(val_prices, val_feats, env_cfg)
        res_val_norm = _eval_env(policy, val_env_val_norm)

        # Lag test: usa features defasadas em 1 barra para ver sensibilidade a timing
        lag_feats = val_feats.shift(1).dropna()
        if not lag_feats.empty:
            lag_prices = val_prices.iloc[val_feats.shape[0] - lag_feats.shape[0] :]
            val_env_lag = BTCMixtureEnv(
                lag_prices,
                lag_feats,
                env_cfg,
                norm_mean=train_norm_mean,
                norm_std=train_norm_std,
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

    summary_path = outdir / "wf_summary.json"
    summary_path.write_text(json.dumps(histories, indent=2))
    _log(f"[WF] Concluído. Resultados em {summary_path}")


if __name__ == "__main__":
    main()
