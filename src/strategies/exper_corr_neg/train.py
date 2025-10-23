from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import load_btc_1h, prepare_dataset
from .env import BTCMixtureEnv, EnvConfig
from .models import MoEPolicy, PPOConfig
from .trainer import PPOTrainer


DEFAULT_CONFIG = Path("src/strategies/exper_corr_neg/config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MoE PPO agent (config-driven)")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to JSON configuration (default: src/strategies/exper_corr_neg/config.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    config = json.loads(cfg_path.read_text())

    df = load_btc_1h(days=3650)
    dataset = prepare_dataset(df)

    price_cols = ["open", "high", "low", "close", "volume"]
    price_df = dataset[price_cols]
    feat_df = dataset.drop(columns=price_cols)

    env_cfg = EnvConfig(**config.get("env", {}))
    env = BTCMixtureEnv(price_df, feat_df, env_cfg)

    input_dim = feat_df.shape[1]
    model_cfg = config.get("model", {})
    policy = MoEPolicy(
        input_dim=input_dim,
        num_actions=3,
        expert_hidden=model_cfg.get("expert_hidden", [64, 32]),
        gating_hidden=model_cfg.get("gating_hidden", [64, 32]),
        num_experts=model_cfg.get("num_experts", 5),
        temperature=model_cfg.get("temperature", 0.7),
        top_k=model_cfg.get("top_k", 2),
    )
    ppo_cfg = PPOConfig(**config.get("ppo", {}))
    train_cfg = config.get("train", {})
    trainer = PPOTrainer(policy, ppo_cfg, device=torch.device(train_cfg.get("device", "cpu")))

    outdir = Path(train_cfg.get("outdir", "reports/exper_corr_neg/train"))
    outdir.mkdir(parents=True, exist_ok=True)

    episodes = int(train_cfg.get("episodes", 500))
    rollout_steps = int(train_cfg.get("rollout_steps", 2048))

    for episode in range(1, episodes + 1):
        metrics = trainer.train_step(env, rollout_steps)
        if episode % 10 == 0:
            ckpt_path = outdir / f"moe_policy_ep{episode}.pt"
            torch.save(policy.state_dict(), ckpt_path)
        if episode % 5 == 0:
            print(f"Episode {episode}: {metrics}")

    final_path = outdir / "moe_policy_final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"Treinamento finalizado. Modelo salvo em {final_path}")


if __name__ == "__main__":
    main()
