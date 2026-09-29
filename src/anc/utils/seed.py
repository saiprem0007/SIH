"""Random-seed utilities for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set global random seed for reproducibility (Python, NumPy, PyTorch).

    Args:
        seed: Integer seed value.  Use the value from ``project.seed``
              in *configs/base.yaml* (default 42).

    Note:
        When PyTorch is not installed the function silently skips the
        torch-specific settings so the rest of the codebase can still
        be used in a lightweight environment.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
