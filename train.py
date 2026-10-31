"""Minimal fakeend trainer for PP."""

from __future__ import annotations

import contextlib
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .fakeend import (
    OnlineFakeEndDataset,
    build_fakeend_val_loader,
    build_offline_train_loader,
    build_offline_val_loader,
    fakeend_loss,
)
from .mapping import generated_circuit
from .model import PP


@dataclass
class TrainConfig:
    stage: str = "fp32"
    d: int = 9
    rounds: int = 9
    channels: int = 128
    p: float = 0.005
    batch_size: int = 512
    epochs: int = 100
    total_samples: int = 5_000_000
    num_workers: int = 4
    val_shots_per_rank: int = 5_000
    val_batch_size: int = 256
    val_seed: int = 12345
    eval_every: int = 500
    patience: int = 30
    min_delta: float = 1e-5
    lr_matrix: float = 3e-4
    lr_scalar: float = 1e-4
    lr_w1a2: float = 5e-5
    weight_decay: float = 3e-3
    warmup_steps: int = 1000
    grad_accum: int = 1
    clip_grad: float = 1.0
    beta_init: float = 1.0
    beta_min: float = 0.0
    beta_decay: float = 0.1
    beta_decay_start_epochs: int = 10
    corr_mwpm: bool = False
    freeze_except_readout_head: bool = False
    train_data_dir: str = ""
    val_data_dir: str = ""
    save_path: str = ""
    resume_path: str = ""
    reset_optimizer: bool = False
    use_amp: bool = True


def setup_dist() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def base_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def split_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    matrix, scalar, w1a2 = [], [], []
    w1a2_param_ids: set[int] = set()
    for module in model.modules():
        if getattr(module, "mode", None) == "w1a2":
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    w1a2_param_ids.add(id(param))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in w1a2_param_ids:
            w1a2.append(param)
            continue
        n = name.lower()
        is_matrix = param.dim() >= 2 and "norm" not in n and "embedding" not in n and "readout" not in n
        (matrix if is_matrix else scalar).append(param)
    return matrix, scalar, w1a2


def freeze_except_readout_head(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if not hasattr(model, "readout") or not hasattr(model.readout, "reduce"):
        raise AttributeError("model.readout.reduce is required for freeze_except_readout_head")
    for param in model.readout.reduce.parameters():
        param.requires_grad = True


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> optim.Optimizer:
    matrix, scalar, w1a2 = split_params(model)
    param_groups = [
        {"params": matrix, "lr": cfg.lr_matrix, "weight_decay": cfg.weight_decay},
        {"params": scalar, "lr": cfg.lr_scalar, "weight_decay": 0.0},
    ]
    if w1a2:
        param_groups.append({"params": w1a2, "lr": cfg.lr_w1a2, "weight_decay": cfg.weight_decay})
    return optim.AdamW(param_groups, fused=torch.cuda.is_available())


def build_scheduler(optimizer: optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_loader(dataset, num_workers: int, pin_memory: bool) -> DataLoader:
    kwargs = dict(batch_size=None, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def split_data_dirs(data_dir: str) -> list[str]:
    return [item.strip() for item in data_dir.split(",") if item.strip()]


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    global_step: int,
    best_acc: float,
    stalls: int,
    beta: float,
    cfg: TrainConfig,
) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": base_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step": global_step,
            "best_acc": best_acc,
            "stalls": stalls,
            "beta": beta,
            "cfg": cfg.__dict__,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    device: torch.device,
    rank: int,
) -> tuple[int, float, int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model_state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_state = ckpt["model_state_dict"]
    else:
        model_state = ckpt
    model_state = {
        remap_checkpoint_key(k): v
        for k, v in model_state.items()
        if isinstance(v, torch.Tensor)
    }
    current = base_model(model).state_dict()
    usable = {
        k: v
        for k, v in model_state.items()
        if k in current and tuple(v.shape) == tuple(current[k].shape)
    }
    skipped = len(model_state) - len(usable)
    missing, unexpected = base_model(model).load_state_dict(usable, strict=False)
    old_stage = (ckpt.get("cfg") or {}).get("stage")
    same_stage = old_stage == cfg.stage
    global_step, best_acc, stalls, beta = 0, -1.0, 0, cfg.beta_init
    if same_stage and not cfg.reset_optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        global_step = int(ckpt.get("global_step", 0))
        best_acc = float(ckpt.get("best_acc", -1.0))
        stalls = int(ckpt.get("stalls", 0))
        beta = float(ckpt.get("beta", cfg.beta_init))
    if rank == 0:
        print(
            f"[resume] {path} | ckpt_stage={old_stage} current_stage={cfg.stage} "
            f"loaded={len(usable)} skipped={skipped} missing={len(missing)} unexpected={len(unexpected)} "
            f"{'state-restored' if same_stage and not cfg.reset_optimizer else 'weights-only'}",
            flush=True,
        )
    return global_step, best_acc, stalls, beta


def remap_checkpoint_key(key: str) -> str:
    key = key.removeprefix("module.")
    if key.startswith("stem."):
        key = "embedding." + key[len("stem."):]
    if key.startswith("embedding.emb_x_hints."):
        key = "embedding.emb_hint." + key[len("embedding.emb_x_hints."):]
    if key.startswith("embedding.stem_norm."):
        key = "embedding.norm." + key[len("embedding.stem_norm."):]
    if key.startswith("embedding.stem_resnet."):
        key = "embedding.mlp." + key[len("embedding.stem_resnet."):]
    if key.startswith("scatter.z_to_grid_hw"):
        key = "scatter.z_hw" + key[len("scatter.z_to_grid_hw"):]
    if key.startswith("scatter.x_to_grid_hw"):
        key = "scatter.x_hw" + key[len("scatter.x_to_grid_hw"):]
    if key.startswith("res2d_blocks."):
        key = "spatial." + key[len("res2d_blocks."):]
    return key


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
    use_amp: bool,
    beta: float,
) -> tuple[float, float, float, float, float, torch.Tensor]:
    was_training = model.training
    model.eval()
    c_all = t_all = c_last = t_last = teacher_dis = 0.0
    final_loss_sum = 0.0
    fakeend_loss_sum = 0.0
    pos_correct = None
    pos_total = 0.0
    for x, y_fake, y_true in loader:
        x = x.to(device, non_blocking=True)
        y_fake = y_fake.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True).view(-1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred, row_logits = model(x, return_row_logits=True)
            final_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, -1], y_true)
            loss = fakeend_loss(pred, y_fake, y_true, beta)
        pred_bin = (pred > 0).float()
        c_all += (pred_bin == y_fake).float().sum().item()
        t_all += pred_bin.numel()
        c_last += (pred_bin[:, -1].view(-1) == y_true).float().sum().item()
        t_last += y_true.numel()
        teacher_dis += (y_fake[:, -1].view(-1) != y_true).float().sum().item()
        row_bin = (row_logits > 0).float()
        if pos_correct is None:
            pos_correct = torch.zeros(row_bin.shape[-1], device=device, dtype=torch.float64)
        pos_correct += (row_bin == y_true[:, None, None]).float().sum(dim=(0, 1)).to(torch.float64)
        pos_total += float(row_bin.shape[0] * row_bin.shape[1])
        final_loss_sum += float(final_loss.item()) * y_true.numel()
        fakeend_loss_sum += float(loss.item()) * y_true.numel()
    model.train(was_training)

    stats = torch.tensor(
        [c_all, t_all, c_last, t_last, teacher_dis, final_loss_sum, fakeend_loss_sum],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    c_all, t_all, c_last, t_last, teacher_dis, final_loss_sum, fakeend_loss_sum = stats.tolist()
    if pos_correct is None:
        pos_acc = torch.empty(0, device=device, dtype=torch.float64)
    else:
        pos_stats = torch.cat([
            pos_correct,
            torch.tensor([pos_total], device=device, dtype=torch.float64),
        ])
        if world_size > 1:
            dist.all_reduce(pos_stats, op=dist.ReduceOp.SUM)
        pos_total_all = float(pos_stats[-1].item())
        pos_acc = (pos_stats[:-1] / max(1.0, pos_total_all)).to(torch.float32)
    return (
        c_all / max(1.0, t_all),
        c_last / max(1.0, t_last),
        teacher_dis / max(1.0, t_last),
        final_loss_sum / max(1.0, t_last),
        fakeend_loss_sum / max(1.0, t_last),
        pos_acc,
    )


def train_cached_pass(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    world_size: int,
    use_amp: bool,
    beta: float,
    clip_grad: float,
) -> tuple[float, int, int]:
    model.train()
    loss_sum = 0.0
    samples = 0
    steps = 0
    for x, y_fake, y_true in loader:
        x = x.to(device, non_blocking=True)
        y_fake = y_fake.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(x)
            loss = fakeend_loss(pred, y_fake, y_true, beta)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.item()) * x.shape[0]
        samples += x.shape[0]
        steps += 1

    stats = torch.tensor([loss_sum, samples], device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_loss, total_samples = stats.tolist()
    return total_loss / max(1.0, total_samples), int(total_samples), steps


def train(cfg: TrainConfig) -> None:
    rank, local_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    circuit = generated_circuit(cfg.d, cfg.rounds, cfg.p)
    model = PP(
        circuit=circuit,
        d=cfg.d,
        rounds=cfg.rounds,
        channels=cfg.channels,
        stage=cfg.stage,
    ).to(device)
    if cfg.freeze_except_readout_head:
        freeze_except_readout_head(base_model(model))
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    steps_per_epoch = max(1, cfg.total_samples // (cfg.batch_size * world_size * max(1, cfg.grad_accum)))
    total_steps = cfg.epochs * steps_per_epoch
    optimizer = build_optimizer(base_model(model), cfg)
    scheduler = build_scheduler(optimizer, cfg.warmup_steps, total_steps)

    global_step, best_acc, stalls, beta = 0, -1.0, 0, cfg.beta_init
    if cfg.resume_path:
        global_step, best_acc, stalls, beta = load_checkpoint(
            cfg.resume_path, model, optimizer, scheduler, cfg, device, rank,
        )

    import multiprocessing as mp
    shared_beta = mp.Value("d", float(beta))
    if cfg.train_data_dir:
        train_loader = build_offline_train_loader(cfg.train_data_dir, cfg.batch_size)
    else:
        train_loader = make_loader(
            OnlineFakeEndDataset(cfg.d, cfg.rounds, cfg.p, cfg.batch_size, rank, shared_beta, cfg.corr_mwpm),
            cfg.num_workers,
            pin_memory,
        )
    if cfg.val_data_dir:
        val_dirs = split_data_dirs(cfg.val_data_dir)
        val_loader_items = [(os.path.basename(path.rstrip("/\\")), build_offline_val_loader(path, cfg.val_batch_size)) for path in val_dirs]
        val_tensors = None
    else:
        val_loader_items = [(
            f"p={cfg.p:g}",
            build_fakeend_val_loader(
            cfg.d, cfg.rounds, cfg.p, cfg.val_shots_per_rank, cfg.val_batch_size, rank, cfg.val_seed, cfg.corr_mwpm,
        ))]
        val_tensors = val_loader_items[0][1].dataset.tensors

    if rank == 0:
        n_params = sum(p.numel() for p in base_model(model).parameters())
        print(
            f"PP | stage={cfg.stage} modes={base_model(model).modes} "
            f"d={cfg.d} rounds={cfg.rounds} C={cfg.channels} params={n_params/1e6:.3f}M",
            flush=True,
        )
        print(
            f"steps={total_steps} ({steps_per_epoch}/epoch) bs/rank={cfg.batch_size} "
            f"world={world_size} beta={beta:.3f} decoder={'corr-mwpm' if cfg.corr_mwpm else 'mwpm'}",
            flush=True,
        )

    iterator = iter(train_loader)
    accum = max(1, cfg.grad_accum)
    micro = 0
    loss_sum = 0.0
    t0 = last_t = time.monotonic()
    last_step = global_step
    beta_decay_start = cfg.beta_decay_start_epochs * steps_per_epoch

    try:
        while global_step < total_steps:
            x, y_fake, y_true = next(iterator)
            x = x.to(device, non_blocking=True)
            y_fake = y_fake.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)

            sync_ctx = model.no_sync() if isinstance(model, DDP) and (micro + 1) % accum != 0 else contextlib.nullcontext()
            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
                    pred = model(x)
                    loss = fakeend_loss(pred, y_fake, y_true, beta) / accum
                loss.backward()
            micro += 1
            loss_sum += float(loss.item()) * accum
            if micro % accum != 0:
                continue

            if cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            epoch_done = global_step // steps_per_epoch
            if global_step % cfg.eval_every != 0:
                continue

            now = time.monotonic()
            it_s = (global_step - last_step) / max(1e-6, now - last_t)
            last_t = now
            last_step = global_step
            val_metrics = []
            for val_name, val_loader in val_loader_items:
                val_metrics.append((val_name, validate(model, val_loader, device, world_size, cfg.use_amp, beta)))

            acc_all, acc_last, teacher_dis, val_loss_final, val_loss_fakeend, pos_acc = val_metrics[0][1]

            if global_step >= beta_decay_start:
                beta = max(cfg.beta_min, beta - cfg.beta_decay)
                shared_beta.value = float(beta)

            if rank == 0:
                ratio = acc_all / max(1e-5, acc_last)
                p_l = 1.0 - max(min(acc_last, 1.0 - 1e-12), 1e-12) ** (1.0 / max(1, cfg.rounds))
                eta = (total_steps - global_step) / max(1e-6, it_s)
                lr_strs = [f"lr_matrix {optimizer.param_groups[0]['lr']:.2e}", 
                           f"lr_scalar {optimizer.param_groups[1]['lr']:.2e}"]
                if len(optimizer.param_groups) > 2:
                    lr_strs.append(f"lr_w1a2 {optimizer.param_groups[2]['lr']:.2e}")
                lr_info = " | ".join(lr_strs)
                print(
                    f"step {global_step:>6} | {lr_info} | "
                    f"beta {beta:.3f} | loss {loss_sum / max(1, cfg.eval_every):.4f} | "
                    f"val_loss_final {val_loss_final:.4f} | val_loss_fakeend {val_loss_fakeend:.4f} | "
                    f"acc_all_t {acc_all:.5f} | acc_last {acc_last:.5f} | ratio {ratio:.4f} | "
                    f"teacher_dis {teacher_dis:.5f} | P_L/cyc {p_l:.3e} | "
                    f"{it_s:.1f} it/s | t {fmt_hms(now - t0)} | ETA {fmt_hms(eta)}",
                    flush=True,
                )
                if pos_acc.numel() > 0:
                    pos_acc_str = ", ".join(
                        f"d{i}={float(v):.5f}" for i, v in enumerate(pos_acc.tolist())
                    )
                    print(f"    [pos-acc] {pos_acc_str}", flush=True)
                for val_name, metrics in val_metrics[1:]:
                    acc_all_i, acc_last_i, teacher_dis_i, val_loss_final_i, val_loss_fakeend_i, pos_acc_i = metrics
                    print(
                        f"    [val:{val_name}] acc_all_t {acc_all_i:.5f} | acc_last {acc_last_i:.5f} | "
                        f"teacher_dis {teacher_dis_i:.5f} | val_loss_final {val_loss_final_i:.4f} | "
                        f"val_loss_fakeend {val_loss_fakeend_i:.4f}",
                        flush=True,
                    )
                    if pos_acc_i.numel() > 0:
                        pos_acc_i_str = ", ".join(
                            f"d{i}={float(v):.5f}" for i, v in enumerate(pos_acc_i.tolist())
                        )
                        print(f"        [pos-acc] {pos_acc_i_str}", flush=True)
                if acc_last >= best_acc + cfg.min_delta:
                    best_acc = acc_last
                    stalls = 0
                    save_checkpoint(cfg.save_path, model, optimizer, scheduler, global_step, best_acc, stalls, beta, cfg)
                    print(f"    [save] best_acc {best_acc:.5f} -> {cfg.save_path}", flush=True)
                else:
                    stalls += 1
                    print(f"    [stall] {stalls}/{cfg.patience}", flush=True)
            loss_sum = 0.0

            stop = torch.tensor([1 if stalls >= cfg.patience else 0], device=device)
            if world_size > 1:
                dist.broadcast(stop, src=0)
            if stop.item():
                break
    finally:
        cleanup_dist()
