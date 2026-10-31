"""PP backbone blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import ChannelGroupNorm
from .quant import SimpleConv1d, SimpleConv2d


class Res2DBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, mode: str):
        super().__init__()
        self.norm1 = ChannelGroupNorm(channels)
        self.conv1 = SimpleConv2d(channels, padding=dilation, dilation=dilation, mode=mode)
        self.norm2 = ChannelGroupNorm(channels)
        self.conv2 = SimpleConv2d(channels, padding=1, dilation=1, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        h = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        h = self.conv1(F.hardtanh(self.norm1(h), -1.0, 1.0))
        h = self.conv2(F.hardtanh(self.norm2(h), -1.0, 1.0))
        h = h.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return x + h


class ResidualCausalTCNBlock(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, int], mode: str):
        super().__init__()
        self.norm1 = ChannelGroupNorm(channels)
        self.norm2 = ChannelGroupNorm(channels)
        self.conv1 = SimpleConv1d(channels, kernel_size=3, dilation=dilations[0], mode=mode)
        self.conv2 = SimpleConv1d(channels, kernel_size=3, dilation=dilations[1], mode=mode)
        self.pad1 = 2 * dilations[0]
        self.pad2 = 2 * dilations[1]

    @staticmethod
    def _causal_conv(conv: SimpleConv1d, x: torch.Tensor, left_pad: int) -> torch.Tensor:
        x = conv.q_input(x)
        x = F.pad(x, (left_pad, 0), value=0.0)
        return F.conv1d(x, conv.q_weight(), None, dilation=conv.dilation)

    @staticmethod
    def _norm_per_t(norm: ChannelGroupNorm, x: torch.Tensor) -> torch.Tensor:
        N, C, T = x.shape
        h = x.transpose(1, 2).reshape(N * T, C)
        h = norm(h)
        return h.reshape(N, T, C).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        h = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        h = self._causal_conv(self.conv1, F.hardtanh(self._norm_per_t(self.norm1, h), -1.0, 1.0), self.pad1)
        h = self._causal_conv(self.conv2, F.hardtanh(self._norm_per_t(self.norm2, h), -1.0, 1.0), self.pad2)
        h = h.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2).contiguous()
        return x + h
