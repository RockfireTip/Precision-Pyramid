"""PP model."""

from __future__ import annotations

from typing import Sequence

import stim
import torch
import torch.nn as nn

from .blocks import Res2DBlock, ResidualCausalTCNBlock
from .embedding import AQ2Embedding, ScatterToGrid
from .mapping import build_aq2_stab_mapping, compute_detector_grid
from .quant import resolve_modes
from .readout import PerTimeReadout


class PP(nn.Module):
    def __init__(
        self,
        circuit: stim.Circuit,
        d: int,
        rounds: int,
        channels: int = 128,
        stage: str = "fp32",
        embed_mode: str = "auto",
        stem_mode: str = "auto",
        backbone_mode: str = "auto",
        readout_mode: str = "auto",
        spatial_dilations: Sequence[int] | None = None,
        tcn_dilations: tuple[int, int] | None = None,
    ):
        super().__init__()
        embed_mode, stem_mode, backbone_mode, readout_mode = resolve_modes(
            stage, embed_mode, stem_mode, backbone_mode, readout_mode,
        )
        grid = compute_detector_grid(circuit, rounds)
        mapping = build_aq2_stab_mapping(circuit, grid)

        self.d = d
        self.rounds = rounds
        self.channels = channels
        self.stage = stage
        self.T = grid.T
        self.grid_H = grid.grid_H
        self.grid_W = grid.grid_W
        self.num_detectors = circuit.num_detectors
        self.modes = (embed_mode, stem_mode, backbone_mode, readout_mode)

        if spatial_dilations is None:
            if d == 9:
                spatial_dilations = (1, 2, 4)
            elif d == 7:
                spatial_dilations = (1, 2, 3)
            elif d in (3, 5):
                spatial_dilations = (1, 2)
            else:
                raise ValueError(f"Unsupported code distance d={d}; expected one of 3, 5, 7, 9.")
        if tcn_dilations is None:
            if d in (7, 9):
                tcn_dilations = (1, 3)
            elif d in (3, 5):
                tcn_dilations = (1, 2)
            else:
                raise ValueError(f"Unsupported code distance d={d}; expected one of 3, 5, 7, 9.")

        self.embedding = AQ2Embedding(mapping, channels, embed_mode, stem_mode)
        self.scatter = ScatterToGrid(mapping)
        self.tcn1 = ResidualCausalTCNBlock(channels, tcn_dilations, backbone_mode)
        self.spatial = nn.ModuleList([
            Res2DBlock(channels, dilation, backbone_mode)
            for dilation in spatial_dilations
        ])
        self.tcn2 = ResidualCausalTCNBlock(channels, tcn_dilations, backbone_mode)
        self.readout = PerTimeReadout(channels, circuit, grid, d, readout_mode)

    def forward(self, x: torch.Tensor, return_row_logits: bool = False):
        emb_z, emb_x = self.embedding(x)
        h = self.scatter(emb_z, emb_x)
        h = self.tcn1(h)
        for layer in self.spatial:
            h = layer(h)
        h = self.tcn2(h)
        return self.readout(h, return_row_logits=return_row_logits)
