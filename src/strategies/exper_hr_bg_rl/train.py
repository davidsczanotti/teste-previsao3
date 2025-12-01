from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from .data import build_dataset
from .env import EnvConfig, RangeVolEnv
from .models import PolicyValueNet, PPOConfig
from .trainer import PPOTrainer


CFG_PATH = Path("src/strategies/exper_hr_bg_rl/config.json")


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_agent(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    env_cfg_dict = config.get("env", {})
    model_cfg = config.get("model", {})
    ppo_cfg_dict = config.get("ppo", {})

    seed = int(train_cfg.get("seed", 42))
    _set_seeds(seed)

    price_df, feat_df, timestamps = build_dataset(config)

    env_cfg = EnvConfig(**env_cfg_dict)
    env = RangeVolEnv(price_df, feat_df, env_cfg, timestamps=timestamps)

    input_dim = feat_df.shape[1] + 1  # features + posição
    num_actions = env.n_actions
    hidden_sizes = model_cfg.get("hidden_sizes", [128, 64])

    device = torch.device(train_cfg.get("device", "cpu"))
    policy = PolicyValueNet(input_dim=input_dim, hidden_sizes=hidden_sizes, num_actions=num_actions)
    ppo_cfg = PPOConfig(**ppo_cfg_dict)
    trainer = PPOTrainer(policy, ppo_cfg, device=device)

    episodes = int(train_cfg.get("episodes", 1000))
    rollout_steps = int(train_cfg.get("rollout_steps", 512))
    log_every = int(train_cfg.get("log_every", 10))
    eval_every = int(train_cfg.get("eval_every", 50))

    outdir = Path(train_cfg.get("outdir", "src/strategies/exper_hr_bg_rl/reports/train"))
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text("episode,policy_loss,value_loss,entropy,avg_reward,sum_reward,greedy_equity\n")

    best_greedy = float("-inf")
    best_path = outdir / "policy_best.pt"

    def _greedy_eval() -> float:
        env_eval = RangeVolEnv(price_df, feat_df, env_cfg, timestamps=timestamps)
        obs_np = env_eval.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        equity = env_cfg.init_equity
        done = False
        while not done:
            with torch.no_grad():
                dist, _ = policy(obs.unsqueeze(0))
                action = torch.argmax(dist.probs, dim=-1)
            next_obs_np, reward, done, info = env_eval.step(int(action.item()))
            equity = float(info.get("equity", equity))
            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
        return equity

    last_metrics: Dict[str, Any] = {}
    for ep in range(1, episodes + 1):
        stats = trainer.train_step(env, rollout_steps)
        last_metrics = stats

        greedy_equity = float("nan")
        if eval_every > 0 and ep % eval_every == 0:
            greedy_equity = _greedy_eval()
            if greedy_equity == greedy_equity and greedy_equity > best_greedy:
                best_greedy = greedy_equity
                torch.save(policy.state_dict(), best_path)

        with metrics_path.open("a") as f:
            f.write(
                f"{ep},"
                f"{stats['policy_loss']},"
                f"{stats['value_loss']},"
                f"{stats['entropy']},"
                f"{stats['avg_reward']},"
                f"{stats['sum_reward']},"
                f"{greedy_equity}\n"
            )

        if log_every > 0 and ep % log_every == 0:
            print(f"[train] episode {ep}: {stats}, greedy_equity={greedy_equity}")

    final_path = outdir / "policy_final.pt"
    torch.save(policy.state_dict(), final_path)

    return {
        "best_greedy": float(best_greedy),
        "final_path": str(final_path),
        "best_path": str(best_path) if best_path.exists() else None,
        "last_metrics": last_metrics,
    }


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text())
    res = train_agent(cfg)
    print("[exper_hr_bg_rl.train] resumo:", res)


if __name__ == "__main__":
    main()

