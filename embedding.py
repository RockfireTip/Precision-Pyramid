"""AQ2 embedding lookup and grid scatter."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mapping import AQ2StabMapping
from .norms import RMSNorm
from .quant import QuantEmbedding, SimpleLinear


class StemMLP(nn.Module):
    def __init__(self, channels: int, mode: str):
        super().__init__()
        self.fc1 = SimpleLinear(channels, 2 * channels, mode=mode)
        self.fc2 = SimpleLinear(2 * channels, channels, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.hardtanh(self.fc1(x), -1.0, 1.0))


class AQ2Embedding(nn.Module):
    def __init__(
        self,
        mapping: AQ2StabMapping,
        channels: int,
        embed_mode: str,
        stem_mode: str,
    ):
        super().__init__()
        self.channels = channels
        self.T = mapping.T
        self.num_z = mapping.num_z
        self.num_x = mapping.num_x
        for name in (
            "gather_z", "valid_z", "z_neighbors", "z_hint_neighbors",
            "gather_x", "valid_x", "x_neighbors", "x_hint_neighbors",
        ):
            self.register_buffer(name, getattr(mapping, name).clone())
        self.emb_space = QuantEmbedding(32, channels, mode=embed_mode)
        self.emb_temp = QuantEmbedding(4, channels, mode=embed_mode)
        self.emb_hint = QuantEmbedding(16, channels, mode=embed_mode)
        self.norm = RMSNorm(channels)
        self.mlp = StemMLP(channels, mode=stem_mode)
        self.register_buffer("time_pe", self._time_encoding(), persistent=False)

    def _time_encoding(self) -> torch.Tensor:
        pos = torch.arange(self.T, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.channels, 2, dtype=torch.float32) * (-math.log(10000.0) / self.channels))
        pe = torch.zeros(1, self.T, 1, self.channels, dtype=torch.float32)
        pe[0, :, 0, 0::2] = torch.sin(pos * div)
        pe[0, :, 0, 1::2] = torch.cos(pos * div[: pe[0, :, 0, 1::2].shape[-1]])
        return pe

    def _indices(
        self,
        det_raw: torch.Tensor,
        gather: torch.Tensor,
        valid: torch.Tensor,
        neighbors: torch.Tensor,
        hints: torch.Tensor,
        num_spatial: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = det_raw.shape[0]
        det_bits = det_raw.long()
        gather = gather.to(det_raw.device)
        valid = valid.view(self.T, num_spatial).to(det_raw.device)
        x = det_bits.gather(1, gather.unsqueeze(0).expand(B, -1)).view(B, self.T, num_spatial)
        x = (x * valid.to(dtype=x.dtype)).long()

        x_pad = F.pad(x, (0, 1), value=0)
        n = x_pad[:, :, neighbors.to(det_raw.device)]
        idx_space = x * 16 + n[..., 0] * 8 + n[..., 1] * 4 + n[..., 2] * 2 + n[..., 3]

        x_t = F.pad(x, (0, 0, 1, 0), value=0)
        idx_temp = x_t[:, 0:self.T, :] * 2 + x

        det_pad = F.pad(det_bits, (0, 1), value=0)
        h = det_pad.gather(1, hints.to(det_raw.device).view(-1).unsqueeze(0).expand(B, -1))
        h = h.view(B, self.T, num_spatial, 4).long()
        idx_hint = h[..., 0] * 8 + h[..., 1] * 4 + h[..., 2] * 2 + h[..., 3]
        return idx_space.long(), idx_temp.long(), idx_hint.long()

    def _embed_one(
        self,
        det_raw: torch.Tensor,
        gather: torch.Tensor,
        valid: torch.Tensor,
        neighbors: torch.Tensor,
        hints: torch.Tensor,
        num_spatial: int,
    ) -> torch.Tensor:
        idx_space, idx_temp, idx_hint = self._indices(det_raw, gather, valid, neighbors, hints, num_spatial)
        emb = self.emb_space(idx_space) + self.emb_temp(idx_temp) + self.emb_hint(idx_hint)
        emb = emb + self.time_pe.to(device=emb.device, dtype=emb.dtype)
        return emb + self.mlp(self.norm(emb))

    def forward(self, det_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._embed_one(det_raw, self.gather_z, self.valid_z, self.z_neighbors, self.z_hint_neighbors, self.num_z)
        x = self._embed_one(det_raw, self.gather_x, self.valid_x, self.x_neighbors, self.x_hint_neighbors, self.num_x)
        return z, x


class ScatterToGrid(nn.Module):
    def __init__(self, mapping: AQ2StabMapping):
        super().__init__()
        self.T = mapping.T
        self.grid_H = mapping.grid_H
        self.grid_W = mapping.grid_W
        self.register_buffer("z_hw", mapping.z_to_grid_hw.clone())
        self.register_buffer("x_hw", mapping.x_to_grid_hw.clone())

    def forward(self, emb_z: torch.Tensor, emb_x: torch.Tensor) -> torch.Tensor:
        B, _, _, C = emb_z.shape
        out = torch.zeros(B, C, self.T, self.grid_H, self.grid_W, device=emb_z.device, dtype=emb_z.dtype)
        z_hw = self.z_hw.to(emb_z.device)
        x_hw = self.x_hw.to(emb_z.device)
        out[:, :, :, z_hw[:, 0], z_hw[:, 1]] = emb_z.permute(0, 3, 1, 2)
        out[:, :, :, x_hw[:, 0], x_hw[:, 1]] = emb_x.permute(0, 3, 1, 2)
        return out
