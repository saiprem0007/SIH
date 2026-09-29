"""
NLMS Post-Filter for ANC-SIH hybrid pipeline.

Sits downstream of the AI enhancement model. Uses the reference channel
to mop up residual correlated noise that the AI model did not fully suppress.

Features
--------
* Multi-tap NLMS (configurable taps 16-128)
* Adaptive step size mu: backed off when SNR is high (convergence is good)
* Divergence protection: weight-norm clip + reset on magnitude explosion
* Double-talk detector (DTD): freezes adaptation when speech energy dominates
  so the filter doesn't try to cancel speech residuals
* Regularisation delta prevents division-by-zero in silent passages

Double-talk detector
--------------------
Compares a short-term energy of the primary *after AI* against a longer-term
background estimate.  When the AI output is "mostly speech" (energy well above
noise floor), DTD fires and adaptation is paused.  This is a simple but
effective causal approach.

Usage (streaming)
-----------------
    filt = NLMSPostFilter(n_taps=64, mu=0.02)
    filt.reset()
    out = filt.process_frame(ai_output_frame, reference_frame)
    # or full signal:
    out = filt.process_signal(ai_output, reference)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EPS: float = 1e-10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NLMSPostFilterConfig:
    n_taps: int = 64              # filter length (16-128 are good ranges)
    mu: float = 0.02              # nominal step size
    delta: float = 1e-5           # regularisation floor
    # Divergence protection
    max_weight_norm: float = 10.0 # if ||w|| exceeds this, reset weights
    # Double-talk detector
    dtd_ratio_threshold: float = 4.0   # speech/noise energy ratio to freeze
    dtd_attack_alpha: float  = 0.90    # short-term energy smoother
    dtd_release_alpha: float = 0.999   # long-term background energy smoother
    # Adaptive mu: scale down when input power is low (convergence regime)
    mu_min_factor: float = 0.1    # mu can go down to mu * mu_min_factor
    mu_max_factor: float = 2.0    # mu can go up to mu * mu_max_factor


# ---------------------------------------------------------------------------
# NLMSPostFilter
# ---------------------------------------------------------------------------

class NLMSPostFilter:
    """Causal NLMS adaptive post-filter.

    Operates on numpy float32.  Fully streaming — state is carried between
    calls to ``process_frame``.

    Args:
        n_taps : filter length in samples
        mu     : nominal normalised step size (< 1 for stability)
        delta  : regularisation constant
        cfg    : optional full config dataclass
    """

    def __init__(
        self,
        n_taps: int = 64,
        mu: float = 0.02,
        delta: float = 1e-5,
        cfg: Optional[NLMSPostFilterConfig] = None,
    ) -> None:
        if cfg is None:
            cfg = NLMSPostFilterConfig(n_taps=n_taps, mu=mu, delta=delta)
        self.cfg = cfg
        self._reset_state()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all filter state (weights, buffers, DTD)."""
        self._reset_state()

    def _reset_state(self) -> None:
        c = self.cfg
        self._w  = np.zeros(c.n_taps, dtype=np.float64)   # filter weights
        self._xb = np.zeros(c.n_taps, dtype=np.float64)   # reference buffer
        # DTD state
        self._st_energy: float  = EPS   # short-term energy of AI output
        self._bg_energy: float  = EPS   # long-term background energy
        self._dtd_frozen: bool  = False
        # Divergence counter
        self._divergence_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_dtd(self, ai_sample: float) -> bool:
        """Update double-talk detector. Returns True if adaptation should freeze."""
        c = self.cfg
        e2 = float(ai_sample ** 2)
        self._st_energy = c.dtd_attack_alpha  * self._st_energy  + (1 - c.dtd_attack_alpha)  * e2
        self._bg_energy = c.dtd_release_alpha * self._bg_energy  + (1 - c.dtd_release_alpha) * e2
        ratio = self._st_energy / (self._bg_energy + EPS)
        self._dtd_frozen = ratio > c.dtd_ratio_threshold
        return self._dtd_frozen

    def _adaptive_mu(self, x_power: float) -> float:
        """Scale mu based on input power (convergence speed control)."""
        c = self.cfg
        # When power is high, use nominal mu; when low, reduce to avoid noise amplification
        norm_power = float(x_power / (np.mean(self._xb ** 2) + EPS))
        scale = float(np.clip(norm_power, c.mu_min_factor, c.mu_max_factor))
        return c.mu * scale

    def _check_divergence(self) -> bool:
        """Return True if weights have diverged; reset if so."""
        w_norm = float(np.linalg.norm(self._w))
        if w_norm > self.cfg.max_weight_norm or not np.isfinite(w_norm):
            logger.warning(
                "NLMS divergence detected (||w||=%.2f). Resetting weights.", w_norm
            )
            self._w[:] = 0.0
            self._divergence_count += 1
            return True
        return False

    # ------------------------------------------------------------------
    # Sample-level update
    # ------------------------------------------------------------------

    def _process_sample(self, d: float, x: float) -> float:
        """Process one sample.

        Args:
            d : desired signal (AI model output, primary channel after AI)
            x : reference signal sample

        Returns:
            e : enhanced output sample (error = d - estimated_noise)
        """
        # Shift buffer (newest at index 0)
        self._xb = np.roll(self._xb, 1)
        self._xb[0] = x

        # Filter output (estimated residual noise)
        y = float(np.dot(self._w, self._xb))

        # Error = desired - estimated
        e = d - y

        # DTD: freeze if speech dominates
        frozen = self._update_dtd(d)

        if not frozen:
            # NLMS update
            x_power = float(np.dot(self._xb, self._xb))
            mu_eff = self._adaptive_mu(x_power)
            denom = x_power + self.cfg.delta
            self._w += (mu_eff / denom) * e * self._xb

            # Divergence check every update
            self._check_divergence()

        return float(e)

    # ------------------------------------------------------------------
    # Frame-level API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        ai_output: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Process one frame (hop_len samples).

        Args:
            ai_output  : (hop_len,) AI model output (noisy residual remains)
            reference  : (hop_len,) reference mic frame

        Returns:
            enhanced   : (hop_len,) float32, NLMS-cleaned output
        """
        if len(ai_output) != len(reference):
            raise ValueError(
                f"Frame length mismatch: ai={len(ai_output)}, ref={len(reference)}"
            )
        d = ai_output.astype(np.float64)
        x = reference.astype(np.float64)
        out = np.empty_like(d)
        for i in range(len(d)):
            out[i] = self._process_sample(d[i], x[i])
        return out.astype(np.float32)

    def process_signal(
        self,
        ai_output: np.ndarray,
        reference: np.ndarray,
        reset: bool = True,
    ) -> np.ndarray:
        """Process a full signal (offline / batch mode).

        Uses the same streaming path as process_frame, so output
        is identical to chunk-by-chunk streaming.

        Args:
            ai_output : (N,) float32 AI model output
            reference : (N,) float32 reference channel
            reset     : if True, reset state before processing

        Returns:
            enhanced  : (N,) float32
        """
        if reset:
            self.reset()
        N = min(len(ai_output), len(reference))
        d = ai_output[:N].astype(np.float64)
        x = reference[:N].astype(np.float64)
        out = np.empty(N, dtype=np.float64)
        for i in range(N):
            out[i] = self._process_sample(d[i], x[i])
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        return {
            "w":           self._w.copy(),
            "xb":          self._xb.copy(),
            "st_energy":   self._st_energy,
            "bg_energy":   self._bg_energy,
            "dtd_frozen":  self._dtd_frozen,
        }

    def set_state(self, state: dict) -> None:
        self._w          = state["w"].copy()
        self._xb         = state["xb"].copy()
        self._st_energy  = float(state["st_energy"])
        self._bg_energy  = float(state["bg_energy"])
        self._dtd_frozen = bool(state["dtd_frozen"])

    @property
    def is_dtd_frozen(self) -> bool:
        """True if the DTD froze adaptation on the last processed sample."""
        return self._dtd_frozen

    @property
    def divergence_count(self) -> int:
        """Number of times the filter has diverged and reset."""
        return self._divergence_count

    @classmethod
    def from_config(cls, cfg) -> "NLMSPostFilter":
        """Build from a pipeline config object (cfg.pipeline.nlms_postfilter.*)."""
        pf = getattr(cfg, "pipeline", None)
        nc = getattr(pf, "nlms_postfilter", None) if pf else None
        if nc is None:
            return cls()
        return cls(
            n_taps=int(nc.n_taps),
            mu=float(nc.mu),
            delta=float(nc.delta),
        )
