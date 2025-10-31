from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


def mlp(input_dim: int, hidden_sizes: List[int], output_dim: int, activation=nn.GELU) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(activation())
        last_dim = size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], num_actions: int) -> None:
        super().__init__()
        self.net = mlp(input_dim, hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden: List[int], temperature: float = 0.7) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_experts = num_experts
        self.net = mlp(input_dim, hidden, num_experts)

    def forward(self, x: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x) / self.temperature
        weights = torch.softmax(logits, dim=-1)
        if top_k >= self.num_experts:
            return weights, torch.ones_like(weights)
        top_values, top_indices = torch.topk(weights, k=top_k, dim=-1)
        mask = torch.zeros_like(weights)
        mask.scatter_(dim=-1, index=top_indices, src=torch.ones_like(top_values))
        masked_weights = weights * mask
        norm_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-9)
        return norm_weights, mask


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    train_iters: int = 10
    batch_size: int = 2048


class MoEPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        expert_hidden: List[int],
        gating_hidden: List[int],
        num_experts: int = 5,
        temperature: float = 0.7,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_hidden, num_actions) for _ in range(num_experts)]
        )
        self.gating = GatingNetwork(input_dim, num_experts, gating_hidden, temperature=temperature)
        self.value_net = mlp(input_dim, expert_hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        expert_logits = torch.stack([expert(obs) for expert in self.experts], dim=1)  # [B, E, A]
        weights, mask = self.gating(obs, top_k=self.top_k)
        mixed_logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        dist = Categorical(logits=mixed_logits)
        value = self.value_net(obs).squeeze(-1)
        # load balancing (variance of weights)
        lb_loss = mask.mean(dim=0).var()  # encourages spread usage
        return dist, value, lb_loss

