"""Per-time row-ensemble fakeend readout."""

from __future__ import annotations

import stim
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mapping import DetectorGrid
from .quant import SimpleConv2d, SimpleLinear


class PerTimeReadout(nn.Module):
    def __init__(
        self,
        channels: int,
        circuit: stim.Circuit,
        grid: DetectorGrid,
        d: int,
        mode: str,
    ):
        super().__init__()
        self.mode = mode
        readout_mode = "w1a4" if mode == "rd_w1a4" else mode
        self.norm = nn.LayerNorm(channels)
        self.scatter = SimpleConv2d(channels, kernel_size=2, padding=0, mode=readout_mode)
        self.fc1 = SimpleLinear(channels, 4 * channels, mode=readout_mode)
        self.fc2 = SimpleLinear(4 * channels, 1, mode="w1a32") if mode == "rd_w1a4" else nn.Linear(4 * channels, 1)
        self.data_rows = d
        self.data_cols = d

    def forward(self, x: torch.Tensor, return_row_logits: bool = False):
        B, C, T, H, W = x.shape
        h = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        h = self.scatter(h)
        h = h.reshape(B, T, C, H - 1, W - 1).permute(0, 2, 1, 3, 4).contiguous()
        if h.shape[-2:] != (self.data_rows, self.data_cols):
            raise RuntimeError(
                f"expected data grid {(self.data_rows, self.data_cols)}, got {tuple(h.shape[-2:])}"
            )
        row_pooled = h.mean(dim=-1)                         # [B, C, T, d]
        row_pooled = row_pooled.permute(0, 2, 3, 1)         # [B, T, d, C]
        z = self.fc1(self.norm(row_pooled))                 # [B, T, d, 4C]
        z = F.relu(z)
        row_logits = self.fc2(z).squeeze(-1)                # [B, T, d]
        logits = row_logits.mean(dim=-1)  # [B, T]
        if return_row_logits:
            return logits, row_logits
        return logits
