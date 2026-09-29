#!/usr/bin/env python
"""
ANC-SIH Phase 1 Training Script.

Usage
-----
# Debug run on CPU (2 minutes, tiny dataset):
python scripts/train.py --config configs/base.yaml --debug

# Full run:
python scripts/train.py --config configs/base.yaml --model convtasnet --variant base

# Resume from checkpoint:
python scripts/train.py --config configs/base.yaml --resume runs/checkpoints/last.pt

Flags
-----
--config PATH       YAML config file (default: configs/base.yaml)
--model NAME        override cfg.model.name (convtasnet | dcunet | lite_stream)
--variant NAME      override cfg.model.variant (base | lite)
--debug             enable debug mode (tiny dataset, 3 epochs, CPU)
--resume PATH       resume from checkpoint file
--no-amp            disable AMP even on CUDA
--seed INT          override random seed
--epochs INT        override number of training epochs
--batch-size INT    override batch size
--lr FLOAT          override learning rate
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from anc.utils.config import load_config, merge_configs, Config
from anc.utils.seed import set_seed
from anc.utils.logging_utils import get_logger
from anc.models import build_model
from anc.losses import CombinedLoss
from anc.train import TrainingEngine, CheckpointManager, WarmupCosineScheduler
from anc.data.dataset import SpeechEnhancementDataset, FixedTestSet, generate_fixed_test_set
from anc.data.manifest import load_manifest, build_synthetic_manifest, merge_manifests, save_manifest
from anc.data.synthetic_dataset import PureSyntheticDataset

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device(cfg) -> torch.device:
    d = str(cfg.project.device).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    oc = cfg.train.optimizer
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(oc.lr),
        weight_decay=float(oc.weight_decay),
        betas=tuple(float(b) for b in oc.betas),
    )


def _build_scheduler(optimizer, cfg, total_epochs: int):
    sc = cfg.train.scheduler
    name = str(sc.name).lower()
    if name == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=int(sc.warmup_epochs),
            total_epochs=total_epochs,
            min_lr=float(sc.min_lr),
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=int(sc.patience),
            factor=float(sc.factor),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def _load_or_create_manifests(cfg, demo: bool = False):
    """Load train/val speech + noise manifests. Fall back to synthetic demo."""
    mdir = Path(cfg.data.manifest_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    def _load_or_synth(path: Path, split: str, kind: str):
        if path.exists():
            return load_manifest(path)
        logger.warning("Manifest %s not found — using synthetic %s.", path, kind)
        df = build_synthetic_manifest(split=split)
        save_manifest(df, path)
        return df

    speech_train = _load_or_synth(mdir / "speech_train.csv", "train", "speech")
    speech_val   = _load_or_synth(mdir / "speech_val.csv",   "val",   "speech")
    noise_train  = _load_or_synth(mdir / "noise_train.csv",  "train", "noise")
    noise_val    = _load_or_synth(mdir / "noise_val.csv",    "val",   "noise")
    return speech_train, speech_val, noise_train, noise_val


def _ensure_testset(cfg):
    """Generate fixed test set if it doesn't exist."""
    tdir = Path(cfg.data.testset_dir)
    manifest = tdir / "manifest.csv"
    if manifest.exists():
        return FixedTestSet(tdir)
    logger.info("Generating fixed test set at %s …", tdir)
    mdir = Path(cfg.data.manifest_dir)
    speech_test = _load_speech_test(mdir)
    noise_test  = _load_noise_test(mdir)
    generate_fixed_test_set(
        cfg, speech_test, noise_test, tdir,
        n_per_tier=5, n_per_category=2, seed=42,
    )
    return FixedTestSet(tdir)


def _load_speech_test(mdir: Path):
    p = mdir / "speech_test.csv"
    if p.exists():
        return load_manifest(p)
    return build_synthetic_manifest(split="test")


def _load_noise_test(mdir: Path):
    p = mdir / "noise_test.csv"
    if p.exists():
        return load_manifest(p)
    return build_synthetic_manifest(split="test")


def _curriculum_snr(cfg, epoch: int):
    """Return (snr_min, snr_max, impulse_prob) for the current curriculum phase."""
    if not cfg.train.curriculum.enabled:
        return float(cfg.data.snr_min), float(cfg.data.snr_max), float(cfg.mixing.impulse_prob)
    for phase in cfg.train.curriculum.phases:
        # phases may be plain dicts (from YAML) or Config objects
        epoch_end = phase["epoch_end"] if isinstance(phase, dict) else int(phase.epoch_end)
        if epoch <= epoch_end:
            if isinstance(phase, dict):
                return float(phase["snr_min"]), float(phase["snr_max"]), float(phase["impulse_prob"])
            return float(phase.snr_min), float(phase.snr_max), float(phase.impulse_prob)
    last = cfg.train.curriculum.phases[-1]
    if isinstance(last, dict):
        return float(last["snr_min"]), float(last["snr_max"]), float(last["impulse_prob"])
    return float(last.snr_min), float(last.snr_max), float(last.impulse_prob)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopper:
    def __init__(self, patience: int, min_delta: float, higher_is_better: bool = True):
        self.patience         = patience
        self.min_delta        = min_delta
        self.higher_is_better = higher_is_better
        self._best            = float("-inf") if higher_is_better else float("inf")
        self._counter         = 0

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        import math
        if math.isnan(value):
            self._counter += 1
            return self._counter >= self.patience
        improved = (
            value > self._best + self.min_delta
            if self.higher_is_better
            else value < self._best - self.min_delta
        )
        if improved:
            self._best    = value
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ANC-SIH Phase 1 Training")
    p.add_argument("--config",     default="configs/base.yaml")
    p.add_argument("--model",      default=None)
    p.add_argument("--variant",    default=None)
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--resume",     default=None)
    p.add_argument("--no-amp",     action="store_true")
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Config ----
    cfg = load_config(args.config)

    # CLI overrides
    overrides = {}
    if args.model:
        overrides.setdefault("model", {})["name"] = args.model
    if args.variant:
        overrides.setdefault("model", {})["variant"] = args.variant
    if args.debug:
        overrides.setdefault("train", {}).setdefault("debug", {})["enabled"] = True
    if args.epochs:
        overrides.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("train", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("train", {}).setdefault("optimizer", {})["lr"] = args.lr
    if overrides:
        cfg = merge_configs(cfg, overrides)

    # Debug mode
    debug = bool(cfg.train.debug.enabled)
    if debug:
        logger.info("DEBUG MODE — tiny dataset, 3 epochs, CPU forced")
        cfg = merge_configs(cfg, {
            "project": {"device": "cpu"},
            "train": {
                "epochs": int(cfg.train.debug.epochs),
                "batch_size": int(cfg.train.debug.batch_size),
                "num_workers": 0,
                "use_amp": False,
                "validate_every_n_epochs": 1,
                "save_audio_every_n_epochs": 1,
            },
        })

    seed = args.seed if args.seed is not None else int(cfg.project.seed)
    set_seed(seed)

    device = _get_device(cfg)
    logger.info("Device: %s", device)

    # ---- Manifests & Datasets ----
    speech_train, speech_val, noise_train, noise_val = _load_or_create_manifests(cfg)

    train_size = int(cfg.train.debug.n_train) if debug else int(cfg.train.virtual_epoch_size)
    val_size   = int(cfg.train.debug.n_val)   if debug else int(cfg.train.val_samples)

    if debug:
        # Use fully-synthetic dataset so --debug works without any audio files
        logger.info("DEBUG: using PureSyntheticDataset (no real files needed)")
        train_ds = PureSyntheticDataset(cfg, split="train", virtual_size=train_size)
        val_ds   = PureSyntheticDataset(cfg, split="val",   virtual_size=val_size)
    else:
        train_ds = SpeechEnhancementDataset(
            cfg, split="train",
            speech_manifest=speech_train,
            noise_manifest=noise_train,
            virtual_size=train_size,
        )
        val_ds = SpeechEnhancementDataset(
            cfg, split="val",
            speech_manifest=speech_val,
            noise_manifest=noise_val,
            virtual_size=val_size,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        drop_last=False,
    )

    # ---- Model ----
    model = build_model(cfg).to(device)
    logger.info("Model: %s", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s (%.2f M)", f"{n_params:,}", n_params / 1e6)

    # ---- Loss ----
    criterion = CombinedLoss.from_config(cfg)

    # ---- Optimiser & Scheduler ----
    total_epochs = int(cfg.train.epochs)
    optimizer  = _build_optimizer(model, cfg)
    scheduler  = _build_scheduler(optimizer, cfg, total_epochs)
    scaler     = (
        torch.cuda.amp.GradScaler()
        if device.type == "cuda" and not args.no_amp and bool(cfg.train.use_amp)
        else None
    )

    # ---- Checkpoint manager ----
    ckpt_dir = Path(cfg.train.checkpoint_dir) / str(cfg.model.name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mgr = CheckpointManager(
        ckpt_dir,
        metric_name=str(cfg.train.early_stop_metric),
        higher_is_better=True,
    )

    # ---- TensorBoard ----
    tb_dir = Path(cfg.train.log_dir) / str(cfg.model.name)
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))

    # ---- Resume ----
    start_epoch  = 1
    global_step  = 0
    if args.resume:
        state = ckpt_mgr.load(model, optimizer, scheduler, which="last", device=device)
        start_epoch = state["epoch"] + 1
        global_step = state.get("global_step", 0)
        logger.info("Resumed from epoch %d", state["epoch"])

    # ---- Early stopping ----
    stopper = EarlyStopper(
        patience=int(cfg.train.early_stop_patience),
        min_delta=float(cfg.train.early_stop_min_delta),
        higher_is_better=True,
    )

    engine = TrainingEngine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        cfg=cfg,
        writer=writer,
        device=device,
        sr=int(cfg.audio.sample_rate),
    )

    audio_dir = Path(cfg.train.log_dir) / "audio_samples"
    val_audio_every = int(cfg.train.save_audio_every_n_epochs)
    validate_every  = int(cfg.train.validate_every_n_epochs)

    logger.info("Starting training: epochs=%d, batch=%d", total_epochs, cfg.train.batch_size)

    for epoch in range(start_epoch, total_epochs + 1):
        # Curriculum SNR — only applicable when using MixingEngine-backed dataset
        snr_min, snr_max, imp_prob = _curriculum_snr(cfg, epoch)
        if hasattr(train_ds, "engine"):
            train_ds.engine.cfg = merge_configs(cfg, {
                "data":   {"snr_min": snr_min, "snr_max": snr_max},
                "mixing": {"impulse_prob": imp_prob},
            })

        # Train
        train_loss, global_step = engine.train_epoch(train_loader, epoch, global_step)

        # Validate
        do_val = (epoch % validate_every == 0) or (epoch == total_epochs)
        val_metrics = {"loss": float("nan"), "si_sdr": float("nan"), "stoi": float("nan")}
        if do_val:
            log_audio = (epoch % val_audio_every == 0)
            val_metrics = engine.validate(val_loader, epoch, global_step, log_audio, audio_dir)

        # LR scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["si_sdr"])
        else:
            scheduler.step()
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        # Checkpoint
        if do_val:
            ckpt_mgr.save(
                model, optimizer, scheduler,
                epoch, global_step,
                {"train_loss": train_loss, **val_metrics},
                cfg_dict=cfg.to_dict() if hasattr(cfg, "to_dict") else None,
            )

        # Early stopping
        if do_val and stopper.step(val_metrics["si_sdr"]):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    writer.close()
    logger.info("Training complete. Best checkpoint: %s/best.pt", ckpt_dir)
    logger.info("Run evaluation: python scripts/evaluate.py --config %s --method neural --ckpt %s/best.pt", args.config, ckpt_dir)


if __name__ == "__main__":
    main()
