"""Normalization layers for PP."""

from __future__ import annotations

import torch
import torch.nn as nn


def pick_num_groups(channels: int, max_groups: int = 32, min_per_group: int = 8) -> int:
    upper = min(max_groups, max(1, channels // min_per_group))
    for groups in range(upper, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ChannelGroupNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(pick_num_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)
