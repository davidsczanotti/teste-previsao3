from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .models import MoEPolicy, PPOConfig


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class PPOTrainer:
    def __init__(self, policy: MoEPolicy, config: PPOConfig, device: torch.device = torch.device("cpu")) -> None:
        self.policy = policy.to(device)
        self.cfg = config
        self.device = device
        self.optim = Adam(self.policy.parameters(), lr=config.learning_rate)

    def collect_rollout(self, env, steps: int) -> Dict[str, torch.Tensor]:
        obs_list, act_list, logp_list, rew_list, val_list, done_list, lb_list = [], [], [], [], [], [], []
        # histograma de uso dos experts (top-k) ao longo do rollout
        gate_hist = np.zeros(self.policy.num_experts, dtype=np.float32)
        obs = torch.tensor(env.reset(), dtype=torch.float32, device=self.device)
        for _ in range(steps):
            dist, value, lb_loss = self.policy(obs.unsqueeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            # coleta do gating (executa novamente só o cabeamento do gate; custo pequeno)
            with torch.no_grad():
                _, mask = self.policy.gating(obs.unsqueeze(0), top_k=self.policy.top_k)
                gate_hist += mask.squeeze(0).detach().cpu().numpy().astype(np.float32)
            next_obs, reward, done, _ = env.step(int(action.item()))
            obs_list.append(obs.cpu().numpy())
            act_list.append(action.cpu().numpy())
            logp_list.append(log_prob.detach().cpu().numpy())
            rew_list.append(reward)
            val_list.append(value.detach().cpu().numpy())
            done_list.append(done)
            lb_list.append(lb_loss.detach().cpu().numpy())

            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            if done:
                obs = torch.tensor(env.reset(), dtype=torch.float32, device=self.device)
        return {
            "obs": torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device),
            "actions": torch.tensor(np.array(act_list), dtype=torch.int64, device=self.device),
            "log_probs": torch.tensor(np.array(logp_list), dtype=torch.float32, device=self.device),
            "rewards": torch.tensor(np.array(rew_list), dtype=torch.float32, device=self.device),
            "values": torch.tensor(np.array(val_list), dtype=torch.float32, device=self.device),
            "dones": torch.tensor(np.array(done_list), dtype=torch.float32, device=self.device),
            "load_balance": torch.tensor(np.array(lb_list), dtype=torch.float32, device=self.device),
            "gate_hist": torch.tensor(gate_hist, dtype=torch.float32, device=self.device),
        }

    def compute_returns(self, rewards, values, dones, gamma, lam):
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
                next_non_terminal = 1.0 - dones[-1]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def make_batches(self, data: Dict[str, torch.Tensor], batch_size: int):
        total = data["obs"].shape[0]
        indices = torch.randperm(total)
        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield {k: v[idx] for k, v in data.items() if k not in {"load_balance"}}

    def train_step(self, env, rollout_steps: int) -> Dict[str, float]:
        batch = self.collect_rollout(env, rollout_steps)
        returns, advantages = self.compute_returns(
            batch["rewards"], batch["values"], batch["dones"], self.cfg.gamma, self.cfg.gae_lambda
        )
        dataset = {
            "obs": batch["obs"],
            "actions": batch["actions"],
            "log_probs": batch["log_probs"],
            "returns": returns,
            "advantages": advantages,
            "values": batch["values"],
        }
        lb_mean = batch["load_balance"].mean().item()
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0

        for _ in range(self.cfg.train_iters):
            for minibatch in self.make_batches(dataset, self.cfg.batch_size):
                dist, value, lb_loss = self.policy(minibatch["obs"])
                new_log_prob = dist.log_prob(minibatch["actions"])
                ratio = torch.exp(new_log_prob - minibatch["log_probs"])
                surr1 = ratio * minibatch["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * minibatch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()
                returns_batch = minibatch["returns"].reshape(-1)
                value_batch = value.reshape(-1)
                if returns_batch.shape != value_batch.shape:
                    min_len = min(len(returns_batch), len(value_batch))
                    returns_batch = returns_batch[:min_len]
                    value_batch = value_batch[:min_len]
                value_loss = (returns_batch - value_batch).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                # load balance regularizer
                loss += 0.01 * lb_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy.item()

        mean_steps = len(batch["rewards"])
        avg_reward = batch["rewards"].mean().item()
        sum_reward = batch["rewards"].sum().item()
        expert_hist = batch["gate_hist"].detach().cpu().numpy()
        expert_hist_sum = float(expert_hist.sum()) if float(expert_hist.sum()) > 0 else 1.0
        expert_usage = (expert_hist / expert_hist_sum).tolist()
        return {
            "policy_loss": policy_loss_total / (self.cfg.train_iters * max(1, mean_steps // self.cfg.batch_size + 1)),
            "value_loss": value_loss_total / (self.cfg.train_iters * max(1, mean_steps // self.cfg.batch_size + 1)),
            "entropy": entropy_total / (self.cfg.train_iters * max(1, mean_steps // self.cfg.batch_size + 1)),
            "load_balance": lb_mean,
            "avg_reward": avg_reward,
            "sum_reward": sum_reward,
            "expert_usage": expert_usage,
        }
