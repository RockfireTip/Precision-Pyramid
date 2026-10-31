"""Simple STE quantization used by PP."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1.0).to(grad_output.dtype)


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


def fake_int_unit(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return x
    x = F.hardtanh(x, -1.0, 1.0)
    qmax = float(2 ** (bits - 1) - 1)
    qmin = -float(2 ** (bits - 1))
    scale = max(qmax, 1.0)
    q = RoundSTE.apply(x * scale).clamp(qmin, qmax)
    return q / scale


def fake_uint_unit(x: torch.Tensor, bits: int, bound: float = 2.0) -> torch.Tensor:
    if bits <= 0:
        return x
    x = torch.clamp(x, 0.0, bound)
    qmax = float(2 ** bits - 1)
    scale = max(qmax / bound, 1.0)
    q = RoundSTE.apply(x * scale).clamp(0.0, qmax)
    return q / scale


def fake_int_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return w
    qmax = float(2 ** (bits - 1) - 1)
    qmin = -float(2 ** (bits - 1))
    scale = w.detach().abs().amax().clamp_min(1e-8) / max(qmax, 1.0)
    q = RoundSTE.apply(w / scale).clamp(qmin, qmax)
    return q * scale


def split_mode(mode: str) -> tuple[str, str]:
    modes = {
        "fp": ("fp", "fp"),
        "w1a32": ("w1", "fp"),
        "w1a4": ("w1", "a4"),
        "rd_w1a4": ("w1", "a4"), # In the readout, all layers except fc2 use W1A4; fc2 uses W1A32.
        "w1a2": ("w1", "a2"),
        "w1a8": ("w1", "a8"),
        "int8": ("int", "a8"),
        "int4": ("int", "a4"),
    }
    return modes[mode]


def resolve_modes(
    stage: str,
    embed: str = "auto",
    stem: str = "auto",
    backbone: str = "auto",
    readout: str = "auto",
) -> tuple[str, str, str, str]:
    modes = {
        "fp32": ("fp", "fp", "fp", "fp"),
        "w1a32": ("fp", "w1a32", "w1a32", "fp"),
        "w1a8": ("w1", "w1a8", "w1a8", "fp"),
        "w1a4": ("w1", "w1a4", "w1a4", "fp"),
        "rd_w1a4": ("w1", "w1a4", "w1a4", "rd_w1a4"),
        "w1a2_s1": ("w1", "w1a2", "w1a4", "rd_w1a4"),
        "w1a2_s2": ("w1", "w1a2", "w1a2", "rd_w1a4"),
    }[stage]
    e, s, b, r = modes
    return (
        e if embed == "auto" else embed,
        s if stem == "auto" else stem,
        b if backbone == "auto" else backbone,
        r if readout == "auto" else readout,
    )


def weight_bits(mode: str) -> int:
    return 4 if mode == "int4" else 8


def quant_act(
    x: torch.Tensor,
    mode: str,
    alpha: Optional[nn.Parameter] = None,
) -> torch.Tensor:
    x = F.hardtanh(x, -1.0, 1.0)
    if mode == "fp":
        return x
    if mode == "a1":
        return sign_ste(x) * (alpha if alpha is not None else 1.0)
    if mode == "a8":
        return fake_int_unit(x, 8)
    if mode == "a4":
        return fake_int_unit(x, 4)
    if mode == "a2":
        return fake_int_unit(x, 2)
    raise ValueError(mode)


def quant_weight(w: torch.Tensor, mode: str, bits: int) -> torch.Tensor:
    if mode == "fp":
        return w
    if mode == "w1":
        return sign_ste(F.hardtanh(w, -1.0, 1.0))
    if mode == "int":
        return fake_int_weight(w, bits)
    raise ValueError(mode)


class SimpleLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: str = "fp",
    ):
        super().__init__()
        _, a_mode = split_mode(mode)
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.alpha_a = nn.Parameter(torch.tensor(1.0)) if a_mode == "a1" else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_mode, a_mode = split_mode(self.mode)
        x = quant_act(x, a_mode, self.alpha_a)
        w = quant_weight(self.weight, w_mode, weight_bits(self.mode))
        return F.linear(x, w, self.bias)


class SimpleConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, mode: str):
        super().__init__()
        _, a_mode = split_mode(mode)
        self.mode = mode
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size))
        self.alpha_a = nn.Parameter(torch.tensor(1.0)) if a_mode == "a1" else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def q_input(self, x: torch.Tensor) -> torch.Tensor:
        _, a_mode = split_mode(self.mode)
        return quant_act(x, a_mode, self.alpha_a)

    def q_weight(self) -> torch.Tensor:
        w_mode, _ = split_mode(self.mode)
        return quant_weight(self.weight, w_mode, weight_bits(self.mode))


class SimpleConv2d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1,
        mode: str = "fp",
    ):
        super().__init__()
        _, a_mode = split_mode(mode)
        self.mode = mode
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size, kernel_size))
        self.alpha_a = nn.Parameter(torch.tensor(1.0)) if a_mode == "a1" else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_mode, a_mode = split_mode(self.mode)
        x = quant_act(x, a_mode, self.alpha_a)
        w = quant_weight(self.weight, w_mode, weight_bits(self.mode))
        return F.conv2d(x, w, None, padding=self.padding, dilation=self.dilation)


class QuantEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, mode: str = "fp"):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.alpha_e = nn.Parameter(torch.tensor(1.0)) if mode == "w1" else None
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(embedding_dim))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self.mode == "fp":
            weight = self.weight
        elif self.mode == "w1":
            weight = sign_ste(F.hardtanh(self.weight, -1.0, 1.0)) * self.alpha_e
        else:
            raise ValueError(self.mode)
        return F.embedding(idx, weight)
