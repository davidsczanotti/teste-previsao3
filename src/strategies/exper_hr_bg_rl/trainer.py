from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
from torch import nn

from .models import PolicyValueNet, PPOConfig


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    last_value: torch.Tensor


class PPOTrainer:
    def __init__(
        self,
        policy: PolicyValueNet,
        cfg: PPOConfig,
        device: torch.device,
    ) -> None:
        self.policy = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)

    def collect_rollout(self, env, steps: int) -> RolloutBatch:
        obs_list: list[torch.Tensor] = []
        act_list: list[torch.Tensor] = []
        logp_list: list[torch.Tensor] = []
        rew_list: list[float] = []
        done_list: list[float] = []
        val_list: list[torch.Tensor] = []

        obs_np = env.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        for _ in range(steps):
            # Coleta de trajetórias sem construir grafo de autograd
            with torch.no_grad():
                dist, value = self.policy(obs.unsqueeze(0))
                action = dist.sample()
                logp = dist.log_prob(action)

            next_obs_np, reward, done, _ = env.step(int(action.item()))

            obs_list.append(obs.detach())
            act_list.append(action.squeeze(0).detach())
            logp_list.append(logp.squeeze(0).detach())
            rew_list.append(float(reward))
            done_list.append(1.0 if done else 0.0)
            val_list.append(value.squeeze(0).detach())

            if done:
                obs_np = env.reset()
            else:
                obs_np = next_obs_np
            obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, last_val = self.policy(obs.unsqueeze(0))
        last_value = last_val.squeeze(0).detach()

        return RolloutBatch(
            obs=torch.stack(obs_list, dim=0),
            actions=torch.stack(act_list, dim=0),
            log_probs=torch.stack(logp_list, dim=0),
            rewards=torch.tensor(rew_list, dtype=torch.float32, device=self.device),
            dones=torch.tensor(done_list, dtype=torch.float32, device=self.device),
            values=torch.stack(val_list, dim=0),
            last_value=last_value,
        )

    def _compute_gae(self, batch: RolloutBatch) -> Dict[str, torch.Tensor]:
        rewards = batch.rewards
        values = batch.values
        dones = batch.dones
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0
        values_ext = torch.cat([values, batch.last_value.unsqueeze(0)], dim=0)

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Trata returns/advantages como constantes no update de PPO
        returns = returns.detach()
        advantages = advantages.detach()
        return {"returns": returns, "advantages": advantages}

    def train_step(self, env, rollout_steps: int) -> Dict[str, float]:
        batch = self.collect_rollout(env, rollout_steps)
        gae_dict = self._compute_gae(batch)
        returns = gae_dict["returns"]
        advantages = gae_dict["advantages"]

        dataset = {
            "obs": batch.obs,
            "actions": batch.actions,
            "log_probs": batch.log_probs,
            "returns": returns,
            "advantages": advantages,
            "values": batch.values,
        }

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        updates = 0

        N = batch.obs.shape[0]
        batch_size = self.cfg.batch_size

        for _ in range(self.cfg.train_iters):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]

                obs_mb = dataset["obs"][batch_idx]
                act_mb = dataset["actions"][batch_idx]
                old_logp_mb = dataset["log_probs"][batch_idx]
                ret_mb = dataset["returns"][batch_idx]
                adv_mb = dataset["advantages"][batch_idx]

                dist, value = self.policy(obs_mb)
                new_logp = dist.log_prob(act_mb)
                ratio = torch.exp(new_logp - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (ret_mb - value).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                updates += 1

        if updates == 0:
            updates = 1

        return {
            "policy_loss": total_policy_loss / updates,
            "value_loss": total_value_loss / updates,
            "entropy": total_entropy / updates,
            "avg_reward": float(batch.rewards.mean().item()),
            "sum_reward": float(batch.rewards.sum().item()),
        }
