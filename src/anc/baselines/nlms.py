"""
Normalized Least Mean Squares (NLMS) adaptive filter.

Used to cancel noise in the primary mic signal using the reference mic.
The reference mic captures mostly noise (with some speech leakage).

Update rule (NLMS):
    y[n]   = w[n]^T x[n]               # estimated noise
    e[n]   = d[n] - y[n]               # enhanced (error) signal
    w[n+1] = w[n] + (mu / (||x[n]||^2 + delta)) * e[n] * x[n]
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_N_TAPS: int = 256
_DEFAULT_MU: float = 0.01
_DEFAULT_DELTA: float = 1e-6


class NLMSFilter:
    """Normalized Least Mean Squares (NLMS) adaptive filter.

    The filter maintains an internal delay-line buffer and a weight vector.
    It can operate sample-by-sample (``process_sample``), frame-by-frame
    (``process_frame``), or over a full offline signal (``process_signal``).

    Args:
        n_taps: Number of filter taps (filter order). Must be > 0.
        mu: Step-size (learning rate). Must satisfy 0 < mu < 2 for stability.
        delta: Regularisation constant to avoid division by zero. Must be > 0.

    Raises:
        ValueError: If any parameter is out of its valid range.
    """

    def __init__(
        self,
        n_taps: int = _DEFAULT_N_TAPS,
        mu: float = _DEFAULT_MU,
        delta: float = _DEFAULT_DELTA,
    ) -> None:
        if n_taps <= 0:
            raise ValueError(f"n_taps must be > 0, got {n_taps}")
        if not (0.0 < mu < 2.0):
            raise ValueError(f"mu must satisfy 0 < mu < 2, got {mu}")
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")

        self.n_taps: int = n_taps
        self.mu: float = mu
        self.delta: float = delta

        # Filter weights (float64 for numerical stability)
        self.w: np.ndarray = np.zeros(n_taps, dtype=np.float64)
        # Circular delay-line input buffer
        self.x_buf: np.ndarray = np.zeros(n_taps, dtype=np.float64)

        logger.debug(
            "NLMSFilter initialised — n_taps=%d, mu=%.4f, delta=%.2e",
            n_taps,
            mu,
            delta,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset filter weights and input buffer to zero.

        Use between independent signals to avoid state leakage.
        """
        self.w[:] = 0.0
        self.x_buf[:] = 0.0
        logger.debug("NLMSFilter state reset.")

    def process_sample(self, d: float, x: float) -> float:
        """Process a single sample and update filter weights.

        Args:
            d: Desired signal sample (primary microphone).
            x: Reference signal sample (reference microphone — mostly noise).

        Returns:
            Enhanced (error) signal sample ``e = d - y``.
        """
        # Shift delay-line: newest sample at index 0
        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = x

        # Filter output (estimated noise)
        y: float = float(np.dot(self.w, self.x_buf))

        # Error / enhanced signal
        e: float = float(d) - y

        # Normalised update
        power: float = float(np.dot(self.x_buf, self.x_buf)) + self.delta
        self.w += (self.mu / power) * e * self.x_buf

        return e

    def process_frame(self, d: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Process one frame of audio sample-by-sample.

        Args:
            d: Primary microphone frame, shape ``(frame_len,)``.
            x: Reference microphone frame, shape ``(frame_len,)``.

        Returns:
            Enhanced frame as ``np.float32`` array of shape ``(frame_len,)``.

        Raises:
            ValueError: If ``d`` and ``x`` have different lengths.
        """
        d = np.asarray(d, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)

        if d.ndim != 1 or x.ndim != 1:
            raise ValueError("d and x must be 1-D arrays.")
        if d.shape[0] != x.shape[0]:
            raise ValueError(
                f"d and x must have the same length, got {d.shape[0]} vs {x.shape[0]}"
            )

        frame_len: int = d.shape[0]
        enhanced: np.ndarray = np.empty(frame_len, dtype=np.float64)

        for i in range(frame_len):
            enhanced[i] = self.process_sample(d[i], x[i])

        return enhanced.astype(np.float32)

    def process_signal(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        reset: bool = True,
    ) -> np.ndarray:
        """Process full signals in offline mode.

        Args:
            primary: Primary (desired + noise) signal, shape ``(N,)``.
            reference: Reference (noise) signal, shape ``(N,)``.
            reset: If ``True`` (default), reset filter state before processing.

        Returns:
            Enhanced signal as ``np.float32`` array, same length as input.

        Raises:
            ValueError: If arrays have different lengths or are not 1-D.
        """
        primary = np.asarray(primary, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)

        if primary.ndim != 1 or reference.ndim != 1:
            raise ValueError("primary and reference must be 1-D arrays.")
        if primary.shape[0] != reference.shape[0]:
            raise ValueError(
                "primary and reference must have the same length, "
                f"got {primary.shape[0]} vs {reference.shape[0]}"
            )

        if reset:
            self.reset()

        n_samples: int = primary.shape[0]
        enhanced: np.ndarray = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            enhanced[i] = self.process_sample(primary[i], reference[i])

        logger.debug("NLMSFilter processed %d samples.", n_samples)
        return enhanced.astype(np.float32)

    def get_state(self) -> dict:
        """Return a copy of the current filter state.

        Returns:
            Dictionary with keys ``'w'`` and ``'x_buf'``, each a copy of the
            corresponding ``np.float64`` array.
        """
        return {
            "w": self.w.copy(),
            "x_buf": self.x_buf.copy(),
        }

    def set_state(self, state: dict) -> None:
        """Restore filter state from a previously saved snapshot.

        Args:
            state: Dictionary with keys ``'w'`` and ``'x_buf'``. Arrays must
                   have length equal to ``self.n_taps``.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If array shapes do not match the filter configuration.
        """
        w = np.asarray(state["w"], dtype=np.float64)
        x_buf = np.asarray(state["x_buf"], dtype=np.float64)

        if w.shape != (self.n_taps,):
            raise ValueError(
                f"w shape mismatch: expected ({self.n_taps},), got {w.shape}"
            )
        if x_buf.shape != (self.n_taps,):
            raise ValueError(
                f"x_buf shape mismatch: expected ({self.n_taps},), got {x_buf.shape}"
            )

        self.w = w.copy()
        self.x_buf = x_buf.copy()
        logger.debug("NLMSFilter state restored.")

    @property
    def convergence_factor(self) -> float:
        """Convergence factor ``mu * (2 - mu)``.

        A value in ``(0, 2)`` indicates the filter is stable. A value closer
        to 1.0 implies faster convergence (but potentially less stability).

        Returns:
            Scalar float in the range ``(0, 2)``.
        """
        return self.mu * (2.0 - self.mu)

    @classmethod
    def from_config(cls, cfg) -> "NLMSFilter":
        """Instantiate an :class:`NLMSFilter` from a config object.

        Expects the configuration to expose attributes at the path
        ``cfg.baselines.nlms.{n_taps, mu, delta}``.

        Args:
            cfg: Configuration object (e.g. an OmegaConf DictConfig or any
                 object with nested attribute access).

        Returns:
            A fully initialised :class:`NLMSFilter`.
        """
        nlms_cfg = cfg.baselines.nlms
        return cls(
            n_taps=int(nlms_cfg.n_taps),
            mu=float(nlms_cfg.mu),
            delta=float(nlms_cfg.delta),
        )
