"""Stim geometry and AQ2 stabilizer mapping for PP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import stim
import torch


def generated_circuit(d: int, rounds: int, p: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )


@dataclass(frozen=True)
class DetectorGrid:
    T: int
    grid_H: int
    grid_W: int
    map_t: torch.Tensor
    map_h: torch.Tensor
    map_w: torch.Tensor
    spatial_x: torch.Tensor
    spatial_y: torch.Tensor
    valid_spatial: torch.Tensor


@dataclass(frozen=True)
class AQ2StabMapping:
    gather_z: torch.Tensor
    valid_z: torch.Tensor
    z_neighbors: torch.Tensor
    z_hint_neighbors: torch.Tensor
    z_to_grid_hw: torch.Tensor
    gather_x: torch.Tensor
    valid_x: torch.Tensor
    x_neighbors: torch.Tensor
    x_hint_neighbors: torch.Tensor
    x_to_grid_hw: torch.Tensor
    T: int
    grid_H: int
    grid_W: int
    num_z: int
    num_x: int
    num_detectors: int


def compute_detector_grid(circuit: stim.Circuit, rounds: int) -> DetectorGrid:
    coords = circuit.get_detector_coordinates()
    times = [c[2] if len(c) == 3 else rounds - 1 for c in coords.values()]
    xs = sorted(set(c[0] for c in coords.values()))
    ys = sorted(set(c[1] for c in coords.values()))
    ts = sorted(set(times))
    x_to_w = {x: i for i, x in enumerate(xs)}
    y_to_h = {y: i for i, y in enumerate(ys)}
    t_to_i = {t: i for i, t in enumerate(ts)}

    map_t, map_h, map_w = [], [], []
    for i in range(circuit.num_detectors):
        c = coords[i]
        map_t.append(t_to_i[c[2] if len(c) == 3 else rounds - 1])
        map_h.append(y_to_h[c[1]])
        map_w.append(x_to_w[c[0]])

    map_t = torch.tensor(map_t, dtype=torch.long)
    map_h = torch.tensor(map_h, dtype=torch.long)
    map_w = torch.tensor(map_w, dtype=torch.long)
    valid_spatial = torch.zeros(len(ys), len(xs), dtype=torch.bool)
    valid_spatial[map_h, map_w] = True

    return DetectorGrid(
        T=len(ts),
        grid_H=len(ys),
        grid_W=len(xs),
        map_t=map_t,
        map_h=map_h,
        map_w=map_w,
        spatial_x=torch.tensor(xs, dtype=torch.float32),
        spatial_y=torch.tensor(ys, dtype=torch.float32),
        valid_spatial=valid_spatial,
    )


def _sorted_hw(t: torch.Tensor) -> list[tuple[int, int]]:
    return sorted((int(h), int(w)) for h, w in t.tolist())


def build_aq2_stab_mapping(circuit: stim.Circuit, grid: DetectorGrid) -> AQ2StabMapping:
    num_detectors = circuit.num_detectors
    valid_hw = set(_sorted_hw(grid.valid_spatial.nonzero()))
    z_hw = {
        (int(grid.map_h[i]), int(grid.map_w[i]))
        for i in range(num_detectors)
        if int(grid.map_t[i]) == grid.T - 1
    }
    x_hw = valid_hw - z_hw
    z_locs = sorted(z_hw)
    x_locs = sorted(x_hw)

    by_thw = {
        (int(grid.map_t[i]), int(grid.map_h[i]), int(grid.map_w[i])): i
        for i in range(num_detectors)
    }

    def gather_for(locs: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        gather = torch.zeros(grid.T, len(locs), dtype=torch.long)
        valid = torch.zeros(grid.T, len(locs), dtype=torch.float32)
        for t in range(grid.T):
            for i, (h, w) in enumerate(locs):
                det = by_thw.get((t, h, w))
                if det is not None:
                    gather[t, i] = det
                    valid[t, i] = 1.0
        return gather.flatten(), valid.flatten()

    def same_neighbors(locs: list[tuple[int, int]]) -> torch.Tensor:
        loc_to_i = {hw: i for i, hw in enumerate(locs)}
        pad = len(locs)
        out = torch.full((len(locs), 4), pad, dtype=torch.long)
        for i, (h, w) in enumerate(locs):
            for slot, (dh, dw) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                out[i, slot] = loc_to_i.get((h + dh, w + dw), pad)
        return out

    def hint_neighbors(src: list[tuple[int, int]], dst: list[tuple[int, int]]) -> torch.Tensor:
        hint = torch.full((grid.T, len(src), 4), num_detectors, dtype=torch.long)
        for src_i, (h, w) in enumerate(src):
            nearest = sorted(dst, key=lambda p: ((p[0] - h) ** 2 + (p[1] - w) ** 2, p[0], p[1]))[:4]
            for t in range(grid.T):
                for slot, (oh, ow) in enumerate(nearest):
                    hint[t, src_i, slot] = by_thw.get((t, oh, ow), num_detectors)
        return hint.view(-1, 4)

    gather_z, valid_z = gather_for(z_locs)
    gather_x, valid_x = gather_for(x_locs)
    return AQ2StabMapping(
        gather_z=gather_z,
        valid_z=valid_z,
        z_neighbors=same_neighbors(z_locs),
        z_hint_neighbors=hint_neighbors(z_locs, x_locs),
        z_to_grid_hw=torch.tensor(z_locs, dtype=torch.long),
        gather_x=gather_x,
        valid_x=valid_x,
        x_neighbors=same_neighbors(x_locs),
        x_hint_neighbors=hint_neighbors(x_locs, z_locs),
        x_to_grid_hw=torch.tensor(x_locs, dtype=torch.long),
        T=grid.T,
        grid_H=grid.grid_H,
        grid_W=grid.grid_W,
        num_z=len(z_locs),
        num_x=len(x_locs),
        num_detectors=num_detectors,
    )


_MEASUREMENT_GATES = {"M", "MX", "MY", "MR", "MRX", "MRY"}


def _flattened_instructions(circuit: stim.Circuit):
    try:
        return circuit.flattened()
    except AttributeError:
        return circuit


def _is_qubit_target(target) -> bool:
    return not (
        bool(getattr(target, "is_combiner", False))
        or bool(getattr(target, "is_measurement_record_target", False))
        or bool(getattr(target, "is_sweep_bit_target", False))
    )


def _target_value(target) -> int:
    return int(getattr(target, "value"))


def _count_mpp_products(targets: Sequence) -> int:
    count = 0
    in_product = False
    for target in targets:
        if bool(getattr(target, "is_combiner", False)):
            count += int(in_product)
            in_product = False
        else:
            in_product = True
    return count + int(in_product)


def derive_logical_data_support(
    circuit: stim.Circuit,
    observable_index: int = 0,
) -> tuple[list[int], set[int]]:
    measurement_qubits: list[Optional[int]] = []
    last_layer: list[int] = []
    support: set[int] = set()

    for inst in _flattened_instructions(circuit):
        name = inst.name
        targets = inst.targets_copy()
        if name in _MEASUREMENT_GATES:
            layer = []
            for target in targets:
                if _is_qubit_target(target):
                    q = _target_value(target)
                    measurement_qubits.append(q)
                    layer.append(q)
            if layer:
                last_layer = layer
        elif name == "MPP":
            measurement_qubits.extend([None] * _count_mpp_products(targets))
        elif name == "OBSERVABLE_INCLUDE":
            args = inst.gate_args_copy()
            obs_index = int(args[0]) if args else 0
            if obs_index != observable_index:
                continue
            n = len(measurement_qubits)
            for target in targets:
                if bool(getattr(target, "is_measurement_record_target", False)):
                    idx = n + _target_value(target)
                    if 0 <= idx < n and measurement_qubits[idx] is not None:
                        support.add(measurement_qubits[idx])

    return sorted(set(last_layer) | support), support


def build_data_grid_support_w_dense(
    circuit: stim.Circuit,
    grid: DetectorGrid,
    d: int,
    observable_index: int = 0,
) -> torch.Tensor:
    _, support = derive_logical_data_support(circuit, observable_index)
    qubit_coords = circuit.get_final_qubit_coordinates()
    data_x = [(a + b) * 0.5 for a, b in zip(grid.spatial_x.tolist(), grid.spatial_x.tolist()[1:])]
    data_y = [(a + b) * 0.5 for a, b in zip(grid.spatial_y.tolist(), grid.spatial_y.tolist()[1:])]

    def nearest(values: list[float], target: float) -> int:
        return min(range(len(values)), key=lambda i: abs(values[i] - target))

    w_dense = torch.zeros(d, d, dtype=torch.float32)
    for q in support:
        x, y = qubit_coords[q][:2]
        w_dense[nearest(data_y, float(y)), nearest(data_x, float(x))] += 1.0
    return w_dense / w_dense.sum().clamp_min(1.0)
