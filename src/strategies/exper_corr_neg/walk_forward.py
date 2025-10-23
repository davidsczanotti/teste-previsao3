from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

    device = torch.device(wf_cfg.get("device", "cpu"))
    histories = []

    for idx, window in enumerate(windows, start=1):
        train_prices = price_df.iloc[window.train_start : window.train_end]
        train_feats = feat_df.iloc[window.train_start : window.train_end]
        val_prices = price_df.iloc[window.val_start : window.val_end]
        val_feats = feat_df.iloc[window.val_start : window.val_end]

        env = BTCMixtureEnv(train_prices, train_feats, env_cfg)
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

        # Evaluate on validation
        val_env = BTCMixtureEnv(val_prices, val_feats, env_cfg)
        obs = torch.tensor(val_env.reset(), dtype=torch.float32, device=device)
        rewards = []
        done = False
        while not done:
            dist, _, _ = policy(obs.unsqueeze(0))
            action = torch.argmax(dist.probs, dim=-1).item()
            next_obs, reward, done, info = val_env.step(action)
            rewards.append(reward)
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        histories.append(
            {
                "window": idx,
                "pnl": float(np.sum(rewards)),
                "equity_end": info.get("equity", 0.0),
            }
        )

        ckpt = outdir / f"policy_window{idx}.pt"
        torch.save(policy.state_dict(), ckpt)

    summary_path = outdir / "wf_summary.json"
    summary_path.write_text(json.dumps(histories, indent=2))
    print(f"Walk-forward concluído. Resultados em {summary_path}")


if __name__ == "__main__":
    main()
