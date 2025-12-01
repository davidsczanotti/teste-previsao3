from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


def mlp(input_dim: int, hidden_sizes: List[int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    train_iters: int = 4
    batch_size: int = 2048


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int], num_actions: int) -> None:
        super().__init__()
        self.shared = mlp(input_dim, hidden_sizes, hidden_sizes[-1])
        self.policy_head = nn.Linear(hidden_sizes[-1], num_actions)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        x = self.shared(obs)
        logits = self.policy_head(x)
        logits = torch.nan_to_num(logits)
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        dist = Categorical(logits=logits)
        value = self.value_head(x).squeeze(-1)
        return dist, value

