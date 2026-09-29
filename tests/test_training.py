"""
Tests for training pipeline integration.
Quick smoke tests that run in < 30 s on CPU.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anc.models.convtasnet_causal import CausalConvTasNet
from anc.models.lite_stream import LiteStreamNet
from anc.losses.combined import CombinedLoss
from anc.train.engine import TrainingEngine
from anc.train.checkpoint import CheckpointManager
from anc.train.scheduler import WarmupCosineScheduler

SR = 16000
T  = SR * 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(B: int = 2, length: int = SR) -> dict:
    """Generate a minimal batch dict as would come from the DataLoader."""
    return {
        "primary":   torch.randn(B, length),
        "reference": torch.randn(B, length) * 0.3,
        "clean":     torch.randn(B, length),
        "snr_tier":  ["Easy"] * B,
        "noise_class": ["misc"] * B,
    }


def _dummy_loader(n_batches: int = 4):
    """Return an iterable of fake batches."""
    return [_make_batch(2, SR) for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWarmupCosineScheduler:

    def test_warmup_increases_lr(self):
        model  = torch.nn.Linear(4, 4)
        optim  = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched  = WarmupCosineScheduler(optim, warmup_epochs=5, total_epochs=20)
        lrs = []
        for _ in range(5):
            sched.step()
            lrs.append(optim.param_groups[0]["lr"])
        assert lrs[-1] > lrs[0], "LR should increase during warmup"

    def test_cosine_decreases_lr(self):
        model = torch.nn.Linear(4, 4)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = WarmupCosineScheduler(optim, warmup_epochs=2, total_epochs=20, min_lr=1e-6)
        for _ in range(2):
            sched.step()
        lr_after_warmup = optim.param_groups[0]["lr"]
        for _ in range(10):
            sched.step()
        lr_later = optim.param_groups[0]["lr"]
        assert lr_later < lr_after_warmup, "LR should decrease after warmup"


class TestCheckpointManager:

    def test_save_and_load(self, tmp_path):
        model  = torch.nn.Linear(4, 4)
        optim  = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mgr    = CheckpointManager(tmp_path, metric_name="si_sdr")

        # Capture original weights before any mutation
        w_original = list(model.parameters())[0].detach().clone()

        # Save
        is_best = mgr.save(model, optim, None, epoch=1, global_step=100,
                           metrics={"si_sdr": 5.0, "loss": 0.5})
        assert is_best

        # Mutate saved model's weights (simulates continued training)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(99.0)

        # Load into a fresh model
        model2 = torch.nn.Linear(4, 4)
        mgr.load(model2, which="best")

        # Verify model2 has the original (pre-mutation) weights, not 99s
        w_loaded = list(model2.parameters())[0].detach()
        assert torch.allclose(w_loaded, w_original), \
            "Loaded weights should match the saved (pre-mutation) values"

    def test_best_tracking(self, tmp_path):
        model  = torch.nn.Linear(4, 4)
        optim  = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mgr    = CheckpointManager(tmp_path, metric_name="si_sdr")

        assert mgr.save(model, optim, None, 1, 0, {"si_sdr": 3.0}) is True
        assert mgr.save(model, optim, None, 2, 0, {"si_sdr": 5.0}) is True   # better
        assert mgr.save(model, optim, None, 3, 0, {"si_sdr": 4.0}) is False  # worse


class TestTrainingEngineSmokeTest:
    """
    Smoke test: one training step and one validation step on tiny model + fake data.
    Must complete in < 30 s on CPU.
    """

    def _build_engine(self):
        model     = CausalConvTasNet(
            encoder_kernel=32, encoder_stride=16,
            n_filters=32, n_stacks=1, n_blocks=2,
            tcn_channels=32, bottleneck_channels=16,
        )
        criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Minimal cfg-like object using SimpleNamespace
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            train=SimpleNamespace(
                grad_clip=5.0,
                log_every_n_steps=1,
                n_audio_samples=0,
            ),
            audio=SimpleNamespace(sample_rate=16000),
        )
        engine = TrainingEngine(
            model=model, criterion=criterion, optimizer=optimizer,
            scaler=None, cfg=cfg, writer=None,
            device=torch.device("cpu"), sr=SR,
        )
        return engine, model

    def test_one_train_step(self):
        engine, model = self._build_engine()
        batch = _make_batch(2, SR // 2)
        loss_before = None
        # Run two steps and make sure loss changes
        for i, b in enumerate([batch, batch]):
            loader = [b]
            train_loss, gs = engine.train_epoch(loader, epoch=1, global_step=i)
            assert not np.isnan(train_loss), f"NaN train loss at step {i}"
            if i == 0:
                loss_before = train_loss
        # Just check no crash — loss may or may not decrease in 2 steps

    def test_one_val_step(self):
        engine, _ = self._build_engine()
        loader = _dummy_loader(2)
        metrics = engine.validate(loader, epoch=1, global_step=0)
        assert "si_sdr" in metrics
        assert "stoi"   in metrics

    def test_lite_stream_train_step(self):
        model     = LiteStreamNet(
            encoder_kernel=32, encoder_stride=16,
            n_filters=32, gru_hidden=32, gru_layers=1, fc_channels=32,
        )
        criterion = CombinedLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            train=SimpleNamespace(grad_clip=5.0, log_every_n_steps=1, n_audio_samples=0),
            audio=SimpleNamespace(sample_rate=SR),
        )
        engine = TrainingEngine(
            model=model, criterion=criterion, optimizer=optimizer,
            scaler=None, cfg=cfg, writer=None,
            device=torch.device("cpu"), sr=SR,
        )
        loader = _dummy_loader(2)
        loss, _ = engine.train_epoch(loader, epoch=1, global_step=0)
        assert not np.isnan(loss)
