from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * (-torch.log(torch.tensor(10000.0, device=t.device)) / half))
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class CondEncoder(nn.Module):
    """
    Simple MLP encoder that maps flattened (L*d) window to a conditioning vector.
    """

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self.net(x_flat)


class CondBlock(nn.Module):
    def __init__(self, d: int, d_cond: int):
        super().__init__()
        self.fc = nn.Linear(d_cond, d * 2)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g, b = self.fc(c).chunk(2, dim=-1)  # FiLM
        return h * torch.sigmoid(g) + b


class EpsModel(nn.Module):
    """
    Predicts noise for y_noisy given timestep and condition vector.

    y_noisy: [B, H, d_y]
    t_idx:   [B] (0..T-1)
    cond_vec:[B, d_cond]
    """

    def __init__(self, d_y: int, d_cond: int, model_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(d_y, model_dim)
        self.time = nn.Linear(128, model_dim)
        self.cond = CondBlock(model_dim, d_cond)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(model_dim, nhead=4, batch_first=True, dim_feedforward=model_dim * 2) for _ in range(n_layers)]
        )
        self.out = nn.Linear(model_dim, d_y)

    def forward(self, y_noisy: torch.Tensor, t_idx: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        h = self.inp(y_noisy)  # [B,H,D]
        te = timestep_embed(t_idx, 128)
        te = self.time(te)  # [B,D]
        h = h + te[:, None, :]
        # Broadcast cond across sequence
        c = cond_vec
        h = self.cond(h, c[:, None, :].expand_as(h))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)

