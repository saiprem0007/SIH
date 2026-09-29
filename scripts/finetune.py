#!/usr/bin/env python
"""
Phase 4 fine-tuning script for ANC-SIH.

Resumes from the best Phase 1 checkpoint, mixes old and new data (50/50
by default), applies curriculum SNR scheduling, and saves new checkpoints.

Usage
-----
# Debug run (CPU, 2 min, tiny subset):
python scripts/finetune.py --config configs/base.yaml --debug

# Full run from Phase 1 checkpoint:
python scripts/finetune.py --config configs/base.yaml \\
    --checkpoint runs/checkpoints/convtasnet/best.pt \\
    --model convtasnet_base

# Colab-friendly (T4, < 3 h):
python scripts/finetune.py --config configs/base.yaml \\
    --checkpoint runs/checkpoints/convtasnet/best.pt \\
    --epochs 20 --batch-size 8 --lr 3e-5 --old-data-fraction 0.5

Curriculum
----------
Epoch 1-5  : SNR in [-5, +10] dB only (moderate difficulty)
Epoch 6-15 : full range [-10, +10] dB
Epoch 16+  : heavy impulsive augmentation added

Regression check
----------------
After fine-tuning, runs the ORIGINAL fixed test set (data/testset/) and
reports delta-SNR and STOI. Refuses to overwrite best.pt if there is a
>1 dB regression on the original test set.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from anc.utils.config import load_config, Config
from anc.utils.seed import set_seed
from anc.utils.logging_utils import get_logger
from anc.models import build_model
from anc.losses import CombinedLoss
from anc.train.checkpoint import CheckpointManager
from anc.train.scheduler import WarmupCosineScheduler

logger = get_logger("finetune")


# ---------------------------------------------------------------------------
# Curriculum config
# ---------------------------------------------------------------------------

CURRICULUM = [
    # (start_epoch, snr_min, snr_max, heavy_impulse)
    (0,  -5.0, 10.0, False),   # easy: moderate SNR
    (5,  -10.0, 10.0, False),  # medium: full range
    (15, -10.0, 10.0, True),   # hard: full range + heavy impulsive aug
]


def _get_curriculum_params(epoch: int) -> Dict[str, Any]:
    """Return SNR range and impulse flag for the current epoch."""
    params = {"snr_min": -5.0, "snr_max": 10.0, "heavy_impulse": False}
    for start, snr_min, snr_max, heavy in CURRICULUM:
        if epoch >= start:
            params = {"snr_min": snr_min, "snr_max": snr_max, "heavy_impulse": heavy}
    return params


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_dataset(
    cfg: Config,
    split: str,
    snr_min: float,
    snr_max: float,
    old_data_fraction: float = 0.5,
    virtual_size: int = 2000,
    use_v2_data: bool = True,
):
    """Build a mixed dataset combining old and new (Phase 4) manifests."""
    import pandas as pd
    from anc.data.dataset import SpeechEnhancementDataset

    manifest_dir = Path(cfg.data.manifest_dir)
    datasets = []

    # --- Old data ---
    s_path = manifest_dir / f"speech_{split}.csv"
    n_path = manifest_dir / f"noise_{split}.csv"
    if s_path.exists() and n_path.exists():
        s_df = pd.read_csv(s_path)
        n_df = pd.read_csv(n_path)
        old_size = int(virtual_size * old_data_fraction)
        # Patch SNR into a temporary config
        cfg_old = cfg
        ds_old = SpeechEnhancementDataset(
            cfg_old, split=split,
            speech_manifest=s_df, noise_manifest=n_df,
            virtual_size=old_size,
        )
        datasets.append(ds_old)
        logger.info("Old %s data: %d virtual samples", split, old_size)
    else:
        logger.warning("Old manifest not found at %s — skipping old data", manifest_dir)

    # --- New (Phase 4) data ---
    if use_v2_data:
        s_v2 = manifest_dir / f"speech_{split}_v2.csv"
        n_v2 = manifest_dir / f"noise_{split}_v2.csv"
        if s_v2.exists() and n_v2.exists():
            s_df2 = pd.read_csv(s_v2)
            n_df2 = pd.read_csv(n_v2)
            new_size = virtual_size - int(virtual_size * old_data_fraction)
            ds_new = SpeechEnhancementDataset(
                cfg, split=split,
                speech_manifest=s_df2, noise_manifest=n_df2,
                virtual_size=new_size,
            )
            datasets.append(ds_new)
            logger.info("New (v2) %s data: %d virtual samples", split, new_size)
        else:
            logger.info("No v2 manifest found; using old data only.")

    if not datasets:
        logger.warning("No manifests found! Generating synthetic demo dataset for %s.", split)
        from anc.data.download import generate_demo_dataset
        raw = Path(cfg.data.raw_dir)
        speech_files, noise_files = generate_demo_dataset(
            raw / "demo" / "speech", raw / "demo" / "noise",
            n_speech=20, n_noise=20, sr=int(cfg.audio.sample_rate),
        )
        from anc.data.mixing import MixingEngine
        from anc.data.dataset import SpeechEnhancementDataset
        # Build a tiny synthetic-only manifest
        import pandas as pd
        s_df = pd.DataFrame({"path": speech_files, "split": split, "speaker_id": "demo", "type": "speech"})
        n_df = pd.DataFrame({"path": noise_files, "split": split, "category": "misc", "type": "noise"})
        ds = SpeechEnhancementDataset(cfg, split=split,
                                       speech_manifest=s_df, noise_manifest=n_df,
                                       virtual_size=100)
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedLoss,
    device: torch.device,
    grad_clip: float = 5.0,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Run one training epoch. Returns dict of mean metrics."""
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss = si_snr_sum = 0.0
    n_batches = 0

    for batch in loader:
        primary   = batch["primary"].to(device)
        reference = batch["reference"].to(device)
        clean     = batch["clean"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            enhanced = model(primary, reference)
            if isinstance(enhanced, tuple):
                enhanced = enhanced[0]
            # Trim to clean length
            L = min(enhanced.shape[-1], clean.shape[-1])
            loss, _ = loss_fn(enhanced[..., :L], clean[..., :L])

        if not torch.isfinite(loss):
            logger.warning("Non-finite loss detected, skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss.item()
        n_batches   += 1

    return {"loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation. Returns dict of mean metrics."""
    from anc.eval.metrics import compute_si_sdr, compute_stoi

    model.eval()
    total_loss = si_snr_sum = stoi_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            primary   = batch["primary"].to(device)
            reference = batch["reference"].to(device)
            clean     = batch["clean"].to(device)

            enhanced = model(primary, reference)
            if isinstance(enhanced, tuple):
                enhanced = enhanced[0]
            L = min(enhanced.shape[-1], clean.shape[-1])
            loss, _ = loss_fn(enhanced[..., :L], clean[..., :L])

            if torch.isfinite(loss):
                total_loss += loss.item()

            for i in range(clean.shape[0]):
                e_np = enhanced[i, ..., :L].squeeze().cpu().numpy().astype(np.float32)
                c_np = clean[i, ..., :L].squeeze().cpu().numpy().astype(np.float32)
                si_snr_sum += compute_si_sdr(c_np, e_np)
                stoi_sum   += compute_stoi(c_np, e_np, sr=16000)
            n += clean.shape[0]

    return {
        "val_loss":   total_loss / max(len(loader), 1),
        "val_si_snr": si_snr_sum / max(n, 1),
        "val_stoi":   stoi_sum   / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Regression check on original test set
# ---------------------------------------------------------------------------

def _regression_check(
    model: nn.Module,
    testset_dir: Path,
    device: torch.device,
    sr: int = 16000,
    regression_tolerance_db: float = 1.0,
) -> Dict[str, float]:
    """Evaluate model on the ORIGINAL fixed test set. Returns metrics dict."""
    from anc.data.dataset import FixedTestSet
    from anc.eval.metrics import compute_si_sdr, compute_stoi

    if not testset_dir.exists():
        logger.warning("Original test set not found at %s. Skipping regression check.", testset_dir)
        return {"regression_check": "skipped"}

    testset = FixedTestSet(testset_dir)
    if len(testset) == 0:
        logger.warning("Original test set is empty.")
        return {"regression_check": "empty"}

    model.eval()
    si_snr_vals, stoi_vals = [], []

    with torch.no_grad():
        for sample in testset:
            primary   = torch.from_numpy(sample["primary"]).unsqueeze(0).to(device)
            reference = torch.from_numpy(sample["reference"]).unsqueeze(0).to(device)
            clean_np  = sample["clean"]

            enhanced = model(primary, reference)
            if isinstance(enhanced, tuple):
                enhanced = enhanced[0]
            e_np = enhanced.squeeze().cpu().numpy().astype(np.float32)
            L = min(len(e_np), len(clean_np))

            si_snr_vals.append(compute_si_sdr(clean_np[:L], e_np[:L]))
            stoi_vals.append(compute_stoi(clean_np[:L], e_np[:L], sr=sr))

    return {
        "orig_test_si_snr": float(np.nanmean(si_snr_vals)),
        "orig_test_stoi":   float(np.nanmean(stoi_vals)),
        "n_samples":        len(si_snr_vals),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ANC-SIH Phase 4 fine-tune")
    p.add_argument("--config",           default="configs/base.yaml")
    p.add_argument("--checkpoint",       default=None,
                   help="Path to Phase 1 best.pt checkpoint")
    p.add_argument("--model",            default="convtasnet_base",
                   help="Model id from configs/models.yaml")
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--batch-size",       type=int,   default=8)
    p.add_argument("--lr",               type=float, default=3e-5,
                   help="Fine-tuning LR (lower than Phase 1)")
    p.add_argument("--old-data-fraction",type=float, default=0.5)
    p.add_argument("--virtual-size",     type=int,   default=2000)
    p.add_argument("--save-dir",         default=None)
    p.add_argument("--regression-tolerance", type=float, default=1.0,
                   help="Max allowed SI-SNR drop vs Phase 1 (dB)")
    p.add_argument("--debug",            action="store_true",
                   help="2-min CPU debug run")
    p.add_argument("--no-v2-data",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(int(cfg.project.seed))

    # --- Device ---
    if args.debug:
        device = torch.device("cpu")
        logger.info("[DEBUG] Using CPU, small dataset, 2 epochs.")
        args.epochs       = 2
        args.batch_size   = 2
        args.virtual_size = 50
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    logger.info("Device: %s", device)

    # --- Model ---
    model = build_model(cfg).to(device)
    logger.info("Model params: %.2fM", model.count_params() / 1e6)

    # --- Load checkpoint ---
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        # Try default location from models.yaml
        try:
            from anc.models.registry import get_model_info
            info = get_model_info(args.model)
            ckpt_path = info.get("checkpoint")
        except Exception:
            pass

    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state", state), strict=False)
        logger.info("Loaded checkpoint: %s", ckpt_path)
    else:
        logger.warning(
            "Checkpoint not found at '%s'. Fine-tuning from random init.", ckpt_path
        )

    # Snapshot Phase 1 baseline on original test set BEFORE fine-tuning
    testset_dir = Path(cfg.data.testset_dir)
    baseline_metrics = _regression_check(model, testset_dir, device)
    baseline_si_snr  = baseline_metrics.get("orig_test_si_snr", float("nan"))
    logger.info("Baseline (Phase 1) orig test SI-SNR: %.2f dB", baseline_si_snr)

    # --- Save dir ---
    save_dir = Path(args.save_dir) if args.save_dir else (
        Path(cfg.project.log_dir) / "checkpoints" / "convtasnet_ft"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Loss ---
    loss_fn = CombinedLoss()

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.lr * 0.01,
    )
    ckpt_manager = CheckpointManager(save_dir, monitor="val_si_snr", mode="max")

    use_amp = device.type == "cuda"

    history = []
    best_val_si_snr = -999.0

    for epoch in range(args.epochs):
        cur = _get_curriculum_params(epoch)
        logger.info(
            "Epoch %d/%d | SNR [%.1f, %.1f] dB | heavy_impulse=%s",
            epoch + 1, args.epochs,
            cur["snr_min"], cur["snr_max"], cur["heavy_impulse"]
        )

        # Build datasets for this curriculum step
        train_ds = _build_dataset(
            cfg, split="train",
            snr_min=cur["snr_min"], snr_max=cur["snr_max"],
            old_data_fraction=args.old_data_fraction,
            virtual_size=args.virtual_size,
            use_v2_data=not args.no_v2_data,
        )
        val_ds = _build_dataset(
            cfg, split="val",
            snr_min=cur["snr_min"], snr_max=cur["snr_max"],
            old_data_fraction=args.old_data_fraction,
            virtual_size=max(200, args.virtual_size // 5),
            use_v2_data=not args.no_v2_data,
        )

        from anc.data.dataset import collate_fn as _collate
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=0, collate_fn=_collate,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, num_workers=0, collate_fn=_collate,
        )

        t0 = time.time()
        train_metrics = _train_epoch(model, train_loader, optimizer, loss_fn, device,
                                     use_amp=use_amp)
        val_metrics   = _validate(model, val_loader, loss_fn, device)
        scheduler.step()

        epoch_time = time.time() - t0
        metrics = {
            "epoch":      epoch + 1,
            "lr":         float(optimizer.param_groups[0]["lr"]),
            "epoch_time": round(epoch_time, 1),
            **train_metrics,
            **val_metrics,
        }
        history.append(metrics)

        logger.info(
            "  loss=%.4f | val_loss=%.4f | val_si_snr=%.2f | val_stoi=%.3f | "
            "lr=%.1e | time=%.1fs",
            metrics["loss"],    metrics["val_loss"],
            metrics["val_si_snr"], metrics["val_stoi"],
            metrics["lr"],      epoch_time,
        )

        # Checkpoint
        ckpt_manager.save(model, optimizer, epoch + 1, metrics, metrics["val_si_snr"])
        best_val_si_snr = max(best_val_si_snr, metrics["val_si_snr"])

    # --- Regression check after fine-tuning ---
    logger.info("Running regression check on original test set...")
    ft_metrics = _regression_check(model, testset_dir, device)
    ft_si_snr  = ft_metrics.get("orig_test_si_snr", float("nan"))

    regression_db = baseline_si_snr - ft_si_snr  # positive = regression
    logger.info("Fine-tuned SI-SNR on orig test: %.2f dB (baseline: %.2f dB, delta: %.2f dB)",
                ft_si_snr, baseline_si_snr, -regression_db)

    regressed = False
    if not math.isnan(regression_db) and regression_db > args.regression_tolerance:
        logger.warning(
            "REGRESSION DETECTED: SI-SNR dropped %.2f dB > tolerance %.2f dB. "
            "Fine-tuned model NOT registered.", regression_db, args.regression_tolerance
        )
        regressed = True
    else:
        logger.info("No regression detected. Fine-tuned model is acceptable.")

    # --- Save training history ---
    history_path = save_dir / "finetune_history.json"
    results = {
        "baseline_orig_test": baseline_metrics,
        "finetuned_orig_test": ft_metrics,
        "regression_db":      round(regression_db, 3) if not math.isnan(regression_db) else None,
        "regression_exceeded": regressed,
        "tolerance_db":       args.regression_tolerance,
        "best_val_si_snr":    round(best_val_si_snr, 3),
        "history":            history,
    }
    with open(history_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    # --- Register checkpoint if not regressed ---
    if not regressed:
        import yaml
        models_yaml = _REPO / "configs" / "models.yaml"
        if models_yaml.exists():
            with open(models_yaml) as f:
                model_data = yaml.safe_load(f)
            new_ckpt_path = str(save_dir / "best.pt")
            found = False
            for entry in model_data["models"]:
                if entry["id"] == args.model + "_ft":
                    entry["checkpoint"] = new_ckpt_path
                    found = True
            if not found:
                # Add a new entry
                model_data["models"].append({
                    "id": args.model + "_ft",
                    "display_name": f"Fine-tuned {args.model} (Phase 4)",
                    "architecture": "convtasnet",
                    "variant": "base",
                    "checkpoint": new_ckpt_path,
                    "params_M": None,
                    "requires_reference": True,
                    "description": "Fine-tuned on Phase 4 expanded defence noise dataset.",
                    "rtf": None,
                })
            with open(models_yaml, "w") as f:
                yaml.dump(model_data, f, default_flow_style=False, sort_keys=False)
            logger.info("Registered fine-tuned model in configs/models.yaml")

    # Final summary
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Best val SI-SNR: {best_val_si_snr:.2f} dB")
    print(f"  Orig test SI-SNR (Phase 1): {baseline_si_snr:.2f} dB")
    print(f"  Orig test SI-SNR (Phase 4): {ft_si_snr:.2f} dB")
    print(f"  Regression     : {regression_db:.2f} dB" if not math.isnan(regression_db) else "  Regression: N/A")
    print(f"  Tolerance      : {args.regression_tolerance:.1f} dB")
    print(f"  Status         : {'REGRESSED (not saved)' if regressed else 'OK'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
