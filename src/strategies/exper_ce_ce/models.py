from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


def mlp(input_dim: int, hidden_sizes: List[int], output_dim: int, dropout: float = 0.0) -> nn.Sequential:
  layers: List[nn.Module] = []
  last = input_dim
  for h in hidden_sizes:
    layers.append(nn.Linear(last, h))
    layers.append(nn.ReLU())
    if dropout > 0.0:
      layers.append(nn.Dropout(dropout))
    last = h
  layers.append(nn.Linear(last, output_dim))
  return nn.Sequential(*layers)


@dataclass
class TrainConfig:
  epochs: int
  batch_size: int
  learning_rate: float
  weight_decay: float
  device: torch.device


class CEClassifier(nn.Module):
  def __init__(self, input_dim: int, hidden_sizes: List[int], dropout: float = 0.0, num_classes: int = 3) -> None:
    super().__init__()
    self.net = mlp(input_dim, hidden_sizes, num_classes, dropout=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

