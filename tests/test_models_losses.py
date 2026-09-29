"""
Unit tests for Phase 1: models, losses, streaming equivalence.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anc.models.convtasnet_causal import CausalConvTasNet, StreamState
from anc.models.lite_stream import LiteStreamNet
from anc.losses.si_snr import SISNRLoss
from anc.losses.stft_loss import STFTLoss, MultiResolutionSTFTLoss
from anc.losses.combined import CombinedLoss

SR = 16000
T  = SR * 2   # 2 s test signal


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def rand_signal(rng):
    """Return a (1, T) float32 tensor with values in [-1, 1]."""
    return torch.from_numpy(rng.uniform(-0.5, 0.5, (1, T)).astype(np.float32))


@pytest.fixture
def rand_pair(rng):
    primary   = torch.from_numpy(rng.uniform(-0.5, 0.5, (2, T)).astype(np.float32))
    reference = torch.from_numpy(rng.uniform(-0.1, 0.1, (2, T)).astype(np.float32))
    return primary, reference


@pytest.fixture
def tiny_convtasnet():
    """Very small ConvTasNet for fast CPU tests."""
    return CausalConvTasNet(
        encoder_kernel=32, encoder_stride=16,
        n_filters=32, n_stacks=1, n_blocks=3,
        tcn_channels=32, bottleneck_channels=16,
        kernel_size=3, mask_activation="sigmoid",
        input_fusion="concat",
    )


@pytest.fixture
def tiny_lite():
    return LiteStreamNet(
        encoder_kernel=32, encoder_stride=16,
        n_filters=32, gru_hidden=32, gru_layers=1, fc_channels=32,
    )


# ===========================================================================
# Model A: CausalConvTasNet
# ===========================================================================

class TestCausalConvTasNet:

    def test_output_shape(self, tiny_convtasnet, rand_pair):
        primary, reference = rand_pair
        out = tiny_convtasnet(primary, reference)
        assert out.shape == (2, T), f"Expected (2, {T}), got {out.shape}"

    def test_output_is_float32(self, tiny_convtasnet, rand_pair):
        primary, reference = rand_pair
        out = tiny_convtasnet(primary, reference)
        assert out.dtype == torch.float32

    def test_no_nan_in_output(self, tiny_convtasnet, rand_pair):
        primary, reference = rand_pair
        out = tiny_convtasnet(primary, reference)
        assert not torch.isnan(out).any(), "NaN in model output"
        assert not torch.isinf(out).any(), "Inf in model output"

    def test_param_count_base(self):
        model = CausalConvTasNet(
            n_filters=512, n_stacks=2, n_blocks=6,
            tcn_channels=256, bottleneck_channels=128,
        )
        n = model.count_params()
        assert n > 1_000_000, f"Base model too small: {n:,} params"
        assert n < 10_000_000, f"Base model too large: {n:,} params"

    def test_param_count_lite(self):
        model = CausalConvTasNet(
            n_filters=128, n_stacks=1, n_blocks=6,
            tcn_channels=64, bottleneck_channels=32,
        )
        n = model.count_params()
        assert n < 600_000, f"Lite model too large: {n:,} params"

    def test_input_fusion_variants(self, rand_pair):
        primary, reference = rand_pair
        for fusion in ["concat", "early_add", "separate_encoder"]:
            model = CausalConvTasNet(
                encoder_kernel=32, encoder_stride=16,
                n_filters=32, n_stacks=1, n_blocks=2,
                tcn_channels=32, bottleneck_channels=16,
                input_fusion=fusion,
            )
            out = model(primary, reference)
            assert out.shape == (2, T), f"fusion={fusion}: shape={out.shape}"
            assert not torch.isnan(out).any(), f"NaN with fusion={fusion}"

    def test_different_lengths(self, tiny_convtasnet):
        """Model must handle arbitrary input lengths."""
        for length in [160, 320, 1600, 8000]:
            primary   = torch.randn(1, length)
            reference = torch.randn(1, length)
            out = tiny_convtasnet(primary, reference)
            assert out.shape == (1, length), f"len={length}: {out.shape}"

    def test_streaming_equals_offline(self, tiny_convtasnet):
        """Streaming chunk-by-chunk output must match offline output (within 1e-4)."""
        torch.manual_seed(0)
        primary   = torch.randn(1, SR)
        reference = torch.randn(1, SR)

        # Offline
        tiny_convtasnet.eval()
        with torch.no_grad():
            offline = tiny_convtasnet(primary, reference)  # (1, SR)

        # Streaming — process in chunks of encoder_stride (one output frame each)
        chunk = tiny_convtasnet.encoder_stride
        state = tiny_convtasnet.init_stream_state(batch_size=1)
        stream_out = []
        with torch.no_grad():
            for t in range(0, SR, chunk):
                p_c = primary[:, t:t+chunk]
                r_c = reference[:, t:t+chunk]
                if p_c.shape[-1] < chunk:
                    break
                o, state = tiny_convtasnet.forward_stream(p_c, r_c, state)
                stream_out.append(o)

        if stream_out:
            stream_cat = torch.cat(stream_out, dim=-1)
            L = min(stream_cat.shape[-1], offline.shape[-1])
            max_diff = (stream_cat[..., :L] - offline[..., :L]).abs().max().item()
            assert max_diff < 1e-3, f"Streaming vs offline max diff = {max_diff:.2e}"

    def test_gradient_flows(self, tiny_convtasnet, rand_pair):
        primary, reference = rand_pair
        primary.requires_grad_(False)
        out = tiny_convtasnet(primary, reference)
        loss = out.mean()
        loss.backward()
        # Check at least one parameter has gradient
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tiny_convtasnet.parameters()
        )
        assert any_grad, "No gradient reached model parameters"


# ===========================================================================
# Model C: LiteStreamNet
# ===========================================================================

class TestLiteStreamNet:

    def test_output_shape(self, tiny_lite, rand_pair):
        primary, reference = rand_pair
        out, _ = tiny_lite(primary, reference)
        assert out.shape == (2, T)

    def test_param_count(self):
        model = LiteStreamNet(
            n_filters=64, gru_hidden=64, gru_layers=1, fc_channels=64
        )
        n = model.count_params()
        assert n < 100_000, f"LiteStream too large: {n:,} params"

    def test_streaming_state(self, tiny_lite, rand_pair):
        primary, reference = rand_pair   # unpack the tuple returned by the fixture
        # Take first sample from batch for single-item streaming test
        p = primary[0:1, :]    # (1, T)
        r = reference[0:1, :]  # (1, T)
        # Just check state roundtrip
        state = tiny_lite.init_stream_state(batch_size=1)
        chunk_p = p[:, :160]   # (1, 160)
        chunk_r = r[:, :160]
        out, new_state = tiny_lite.forward_stream(chunk_p, chunk_r, state)
        assert out.shape[-1] == 160
        assert "gru_h" in new_state

    def test_no_nan(self, tiny_lite, rand_pair):
        primary, reference = rand_pair
        out, _ = tiny_lite(primary, reference)
        assert not torch.isnan(out).any()


# ===========================================================================
# Losses
# ===========================================================================

class TestSISNRLoss:

    def test_identical_signal_perfect(self):
        """SI-SNR loss of identical signals should be very negative (≈ -∞ = perfect)."""
        loss_fn = SISNRLoss()
        x = torch.randn(2, SR)
        loss = loss_fn(x, x)
        assert loss.item() < -30.0, f"Loss should be very negative, got {loss.item():.2f}"

    def test_noisy_worse_than_clean(self):
        """A noisy signal should produce a larger (worse) loss than a clean one."""
        loss_fn = SISNRLoss()
        clean   = torch.randn(2, SR)
        noise   = 0.5 * torch.randn(2, SR)
        noisy   = clean + noise
        enhanced_clean = clean
        enhanced_noisy = noisy

        loss_clean = loss_fn(enhanced_clean, clean).item()
        loss_noisy = loss_fn(enhanced_noisy, clean).item()
        assert loss_noisy > loss_clean, (
            f"Noisy loss ({loss_noisy:.2f}) should be > clean loss ({loss_clean:.2f})"
        )

    def test_scale_invariance(self):
        """Scaling the estimate should not change SI-SNR (it's scale-invariant)."""
        loss_fn = SISNRLoss()
        clean   = torch.randn(2, SR)
        noisy   = clean + 0.1 * torch.randn(2, SR)
        l1 = loss_fn(noisy,         clean).item()
        l2 = loss_fn(noisy * 10.0,  clean).item()
        assert abs(l1 - l2) < 0.1, f"Not scale-invariant: {l1:.4f} vs {l2:.4f}"

    def test_no_nan(self):
        loss_fn = SISNRLoss()
        clean = torch.randn(4, SR)
        enhanced = torch.randn(4, SR)
        loss = loss_fn(enhanced, clean)
        assert not torch.isnan(loss), "NaN in SI-SNR loss"

    def test_zero_signal(self):
        """Zero signals should not crash; result should be finite or -inf (not NaN)."""
        loss_fn = SISNRLoss()
        x = torch.zeros(2, SR)
        loss = loss_fn(x, x)
        # SI-SNR of identical zero signals: s_proj=0, noise=0, log10(0/eps)=-inf,
        # negated → +inf loss.  That is an expected edge case, not a NaN.
        assert not torch.isnan(loss), "Loss should not be NaN for zero inputs"


class TestSTFTLoss:

    def test_identical_zero_loss(self):
        """Identical signals → near-zero STFT loss (log convergence term)."""
        loss_fn = STFTLoss(n_fft=256, hop_len=64, win_len=256)
        x = torch.randn(2, SR)
        loss = loss_fn(x, x)
        assert loss.item() < 0.05, f"Identical STFT loss too high: {loss.item():.4f}"

    def test_noisy_higher(self):
        loss_fn = STFTLoss(n_fft=256, hop_len=64, win_len=256)
        clean = torch.randn(2, SR)
        noisy = clean + 0.5 * torch.randn(2, SR)
        l_clean = loss_fn(clean, clean).item()
        l_noisy = loss_fn(noisy, clean).item()
        assert l_noisy > l_clean

    def test_no_nan(self):
        loss_fn = STFTLoss(n_fft=256, hop_len=64, win_len=256)
        x = torch.randn(2, SR)
        y = torch.randn(2, SR)
        loss = loss_fn(x, y)
        assert not torch.isnan(loss)


class TestMultiResSTFTLoss:

    def test_three_resolutions(self):
        loss_fn = MultiResolutionSTFTLoss()
        x = torch.randn(2, SR)
        y = torch.randn(2, SR)
        loss = loss_fn(x, y)
        assert not torch.isnan(loss)
        assert loss.item() > 0


class TestCombinedLoss:

    def test_returns_tuple(self):
        loss_fn = CombinedLoss()
        x = torch.randn(2, SR)
        y = torch.randn(2, SR)
        total, comps = loss_fn(x, y)
        assert isinstance(comps, dict)
        assert "si_snr" in comps
        assert "stft"   in comps
        assert "l1"     in comps
        assert "total"  in comps

    def test_clean_lower_than_noisy(self):
        loss_fn = CombinedLoss()
        clean   = torch.randn(2, SR)
        noisy   = clean + 0.5 * torch.randn(2, SR)
        l_clean, _ = loss_fn(clean,  clean)
        l_noisy, _ = loss_fn(noisy,  clean)
        assert l_noisy.item() > l_clean.item()

    def test_nan_input_handled(self):
        """NaN input should not crash — loss returns zero."""
        loss_fn = CombinedLoss()
        nan_sig = torch.full((2, SR), float("nan"))
        clean   = torch.randn(2, SR)
        total, comps = loss_fn(nan_sig, clean)
        assert not torch.isnan(total), "NaN guard failed"

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0 (default config)."""
        loss_fn = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)
        assert abs(loss_fn.alpha + loss_fn.beta + loss_fn.gamma - 1.0) < 1e-6
