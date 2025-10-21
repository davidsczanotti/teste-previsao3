from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class DiffusionConfig:
    timesteps: int = 50  # T
    beta_start: float = 1e-4
    beta_end: float = 0.02


class DiffusionSchedule:
    def __init__(self, cfg: DiffusionConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._build()

    def _build(self):
        T = self.cfg.timesteps
        betas = torch.linspace(self.cfg.beta_start, self.cfg.beta_end, T, device=self.device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

    def sample_t(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.cfg.timesteps, (batch_size,), device=self.device, dtype=torch.long)


def training_step(model: nn.Module, sched: DiffusionSchedule, optimizer: torch.optim.Optimizer, y: torch.Tensor, cond_vec: torch.Tensor) -> float:
    """
    y: [B,H,1] clean targets
    cond_vec: [B,d_cond]
    """
    device = next(model.parameters()).device
    B = y.shape[0]
    t = sched.sample_t(B)
    noise = torch.randn_like(y)
    alpha_bar_t = sched.alpha_bar[t].view(B, 1, 1)
    y_t = torch.sqrt(alpha_bar_t) * y + torch.sqrt(1 - alpha_bar_t) * noise
    pred_noise = model(y_t, t, cond_vec)
    loss = torch.mean((noise - pred_noise) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def sample(model: nn.Module, sched: DiffusionSchedule, cond_vec: torch.Tensor, horizon: int, d_y: int = 1, ddim: bool = False) -> torch.Tensor:
    """
    Returns shape [B,H,d_y] sampled sequences of future log-returns.
    """
    device = next(model.parameters()).device
    B = cond_vec.shape[0]
    y_t = torch.randn(B, horizon, d_y, device=device)
    T = sched.cfg.timesteps
    for ti in reversed(range(T)):
        t = torch.full((B,), ti, device=device, dtype=torch.long)
        beta_t = sched.betas[ti]
        alpha_t = sched.alphas[ti]
        alpha_bar_t = sched.alpha_bar[ti]
        # predict noise
        eps = model(y_t, t, cond_vec)
        # Compute the mean of p(y_{t-1} | y_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (y_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps)
        if ti > 0:
            if ddim:
                # DDIM: deterministic step (no noise)
                y_t = mean
            else:
                noise = torch.randn_like(y_t)
                sigma_t = torch.sqrt(beta_t)
                y_t = mean + sigma_t * noise
        else:
            y_t = mean
    return y_t

