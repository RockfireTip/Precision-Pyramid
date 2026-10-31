"""Fakeend data and loss utilities."""

from __future__ import annotations

import math
import os
from pathlib import Path
import time

import numpy as np
import stim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from .mapping import generated_circuit


def build_time_masks(circuit: stim.Circuit, rounds: int) -> list[np.ndarray]:
    coords = circuit.get_detector_coordinates()
    det_times = np.array([
        coords[i][2] if len(coords[i]) == 3 else rounds - 1
        for i in range(circuit.num_detectors)
    ])
    return [(det_times <= t).astype(np.uint8) for t in sorted(set(det_times))]


def build_matcher(circuit: stim.Circuit, corr_mwpm: bool):
    import pymatching

    if corr_mwpm:
        return pymatching.Matching.from_stim_circuit(circuit, enable_correlations=True)
    return pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )


class OfflineFakeEndDataset(IterableDataset):
    def __init__(self, data_dir: str, batch_size: int, infinite: bool = True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.infinite = infinite
        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

    def _iter_once(self):
        for file_path in self.files:
            with np.load(file_path) as data:
                x = data["X"]
                y_fake = data["Y_fake"]
                y_true = data["Y_true"]
                n = x.shape[0]
                for start in range(0, n, self.batch_size):
                    end = min(start + self.batch_size, n)
                    yield (
                        torch.from_numpy(x[start:end]).float(),
                        torch.from_numpy(y_fake[start:end]).float(),
                        torch.from_numpy(y_true[start:end]).float(),
                    )

    def __iter__(self):
        while True:
            yielded = False
            for batch in self._iter_once():
                yielded = True
                yield batch
            if not self.infinite or not yielded:
                break


class OnlineFakeEndDataset(IterableDataset):
    def __init__(
        self,
        d: int,
        rounds: int,
        p: float,
        batch_size: int,
        rank: int = 0,
        shared_beta=None,
        corr_mwpm: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.rank = rank
        self.shared_beta = shared_beta
        self.corr_mwpm = corr_mwpm
        self.circuit = generated_circuit(d, rounds, p)
        self.time_masks = build_time_masks(self.circuit, rounds)
        self.matcher = build_matcher(self.circuit, corr_mwpm)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        seed = (time.time_ns() + self.rank * 100000 + worker_id * 10000 + int.from_bytes(os.urandom(4), "little")) % (2**32 - 1)
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        while True:
            det, obs = sampler.sample(self.batch_size, separate_observables=True)
            y_fake = np.zeros((self.batch_size, len(self.time_masks)), dtype=np.float32)
            if self.shared_beta is None or float(self.shared_beta.value) > 0.0:
                for t, mask in enumerate(self.time_masks):
                    y_fake[:, t] = self.matcher.decode_batch(
                        det * mask,
                        enable_correlations=self.corr_mwpm,
                    )[:, 0]
            yield (
                torch.from_numpy(det).float(),
                torch.from_numpy(y_fake).float(),
                torch.from_numpy(obs).float(),
            )


def make_fakeend_val_cache(
    d: int,
    rounds: int,
    p: float,
    n_shots: int,
    rank: int = 0,
    seed: int = 12345,
    corr_mwpm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    circuit = generated_circuit(d, rounds, p)
    sampler = circuit.compile_detector_sampler(seed=(seed + rank * 9999) & 0xFFFFFFFF)
    matcher = build_matcher(circuit, corr_mwpm)
    masks = build_time_masks(circuit, rounds)
    det, obs = sampler.sample(n_shots, separate_observables=True)
    y_fake = [
        torch.from_numpy(
            matcher.decode_batch(det * mask, enable_correlations=corr_mwpm)
        ).float()
        for mask in masks
    ]
    return (
        torch.from_numpy(det).float(),
        torch.stack(y_fake, dim=1).squeeze(-1),
        torch.from_numpy(obs).float(),
    )


def build_fakeend_val_loader(
    d: int,
    rounds: int,
    p: float,
    n_shots: int,
    batch_size: int,
    rank: int = 0,
    seed: int = 12345,
    corr_mwpm: bool = False,
) -> DataLoader:
    return DataLoader(
        TensorDataset(*make_fakeend_val_cache(d, rounds, p, n_shots, rank, seed, corr_mwpm)),
        batch_size=batch_size,
        shuffle=False,
    )


def build_offline_val_loader(data_dir: str, batch_size: int) -> DataLoader:
    return DataLoader(
        OfflineFakeEndDataset(data_dir, batch_size, infinite=False),
        batch_size=None,
    )


def build_offline_train_loader(data_dir: str, batch_size: int) -> DataLoader:
    return DataLoader(
        OfflineFakeEndDataset(data_dir, batch_size, infinite=True),
        batch_size=None,
    )


def cosine_s_weights(T: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(T - 1, dtype=torch.float32, device=device)
    return 0.5 * (1.0 - torch.cos(math.pi * (t + 1.0) / T))


def fakeend_loss(
    pred: torch.Tensor,
    y_fake: torch.Tensor,
    y_true: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    y_true = y_true.view(-1)
    loss_final = F.binary_cross_entropy_with_logits(pred[:, -1], y_true)
    if beta <= 0.0:
        return loss_final
    w = cosine_s_weights(pred.size(1), pred.device).view(1, -1)
    bce = F.binary_cross_entropy_with_logits(pred[:, :-1], y_fake[:, :-1], reduction="none")
    return loss_final + beta * (bce * w).sum(dim=1).mean() / w.sum().clamp_min(1e-6)
