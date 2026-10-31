"""PP training CLI."""

from __future__ import annotations

import argparse
import time

from .train import TrainConfig, train


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="PP fakeend trainer")
    p.add_argument("--stage", choices=["fp32", "w1a32", "w1a8", "w1a4", "w1a2", "w1a2_s1", "w1a2_s2", "rd_w1a4"], default="w1a2_s2")
    p.add_argument("--d", type=int, default=9)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--p", type=float, default=0.003)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--total-samples", type=int, default=5_000_000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-shots-per-rank", type=int, default=5_000)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--val-seed", type=int, default=12345)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--lr-matrix", type=float, default=3e-4)
    p.add_argument("--lr-scalar", type=float, default=1e-4)
    p.add_argument("--lr-w1a2", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=3e-3)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--beta-init", type=float, default=1.0)
    p.add_argument("--beta-min", type=float, default=0.0)
    p.add_argument("--beta-decay", type=float, default=0.1)
    p.add_argument("--beta-decay-start-epochs", type=int, default=10)
    p.add_argument("--corr-mwpm", action="store_true")
    p.add_argument("--freeze-except-readout-head", action="store_true")
    p.add_argument("--train-data-dir", type=str, default="")
    p.add_argument("--val-data-dir", type=str, default="")
    p.add_argument("--save", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--reset-optimizer", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    if args.rounds is None:
        args.rounds = args.d
    if not args.save:
        args.save = f"ckpts/tcn_v5_{args.stage}_d{args.d}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    return TrainConfig(
        stage=args.stage,
        d=args.d,
        rounds=args.rounds,
        channels=args.channels,
        p=args.p,
        batch_size=args.batch_size,
        epochs=args.epochs,
        total_samples=args.total_samples,
        num_workers=args.num_workers,
        val_shots_per_rank=args.val_shots_per_rank,
        val_batch_size=args.val_batch_size,
        val_seed=args.val_seed,
        eval_every=args.eval_every,
        patience=args.patience,
        min_delta=args.min_delta,
        lr_matrix=args.lr_matrix,
        lr_scalar=args.lr_scalar,
        lr_w1a2=args.lr_w1a2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_accum=args.grad_accum,
        clip_grad=args.clip_grad,
        beta_init=args.beta_init,
        beta_min=args.beta_min,
        beta_decay=args.beta_decay,
        beta_decay_start_epochs=args.beta_decay_start_epochs,
        corr_mwpm=args.corr_mwpm,
        freeze_except_readout_head=args.freeze_except_readout_head,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        save_path=args.save,
        resume_path=args.resume,
        reset_optimizer=args.reset_optimizer,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    train(parse_args())
