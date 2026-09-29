"""
Model registry loader for ANC-SIH.

Loads configs/models.yaml and provides:

    get_model_info(model_id)   -> dict
    list_models()              -> List[dict]
    load_hybrid(model_id, ...)  -> HybridANC   (lazy, cached)

RTF benchmarking
----------------
    python scripts/benchmark_rtf.py

fills the "rtf" field in configs/models.yaml automatically.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "configs" / "models.yaml"
_CACHE: Dict[str, Any] = {}   # model_id -> HybridANC


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> List[Dict[str, Any]]:
    with open(_REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    return data["models"]


def list_models() -> List[Dict[str, Any]]:
    """Return all model entries from the registry."""
    return _load_registry()


def get_model_info(model_id: str) -> Dict[str, Any]:
    """Return registry entry for a given model_id.

    Raises:
        KeyError: if model_id is not found
    """
    for entry in _load_registry():
        if entry["id"] == model_id:
            return entry
    raise KeyError(f"Model '{model_id}' not found in registry. "
                   f"Available: {[e['id'] for e in _load_registry()]}")


# ---------------------------------------------------------------------------
# HybridANC loader
# ---------------------------------------------------------------------------

def load_hybrid(
    model_id: str,
    ref_mode: str = "provided",
    device: Optional[str] = None,
    cfg=None,
    force_reload: bool = False,
) -> "HybridANC":  # type: ignore[name-defined]
    """Load a ready-to-use HybridANC for a given model_id.

    Models are lazily loaded and cached in-process (one per model_id).
    Pass force_reload=True to reload from disk.

    Args:
        model_id    : id from configs/models.yaml
        ref_mode    : "provided" | "simulated" | "none"
        device      : "cpu" | "cuda" | None (auto-select)
        cfg         : Config object (loads base.yaml if None)
        force_reload: bypass in-process cache

    Returns:
        HybridANC instance
    """
    cache_key = f"{model_id}::{ref_mode}"
    if not force_reload and cache_key in _CACHE:
        logger.debug("Registry cache hit for '%s'", cache_key)
        return _CACHE[cache_key]

    info = get_model_info(model_id)

    # Auto device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Load config
    if cfg is None:
        from anc.utils.config import load_config
        cfg = load_config(str(Path(__file__).resolve().parent.parent.parent / "configs" / "base.yaml"))

    # Build model
    ai_model = _build_model(info, cfg, device)

    # Build HybridANC
    from anc.pipeline.hybrid import HybridANC
    engine = HybridANC.from_config(cfg, model=ai_model, ref_mode=ref_mode)

    _CACHE[cache_key] = engine
    logger.info("Loaded HybridANC[%s] (ref_mode=%s, device=%s)", model_id, ref_mode, device)
    return engine


def _build_model(info: Dict[str, Any], cfg, device: str):
    """Construct and optionally load a checkpoint for the given registry entry."""
    import torch
    from anc.models import build_model
    from anc.utils.config import merge_configs

    arch = info["architecture"]
    variant = info.get("variant", "base")
    requires_ref = info.get("requires_reference", True)

    # Override config for this variant
    overrides: Dict[str, Any] = {
        "model": {"name": arch, "variant": variant}
    }
    if not requires_ref:
        overrides["model"]["input_fusion"] = "early_add"

    model_cfg = merge_configs(cfg, overrides)
    model = build_model(model_cfg).to(device)

    # Load checkpoint if it exists
    ckpt_path = Path(info.get("checkpoint", ""))
    if not ckpt_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent.parent
        ckpt_path = repo_root / ckpt_path

    if ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location=device)
        model_state = state.get("model_state", state)
        model.load_state_dict(model_state, strict=False)
        logger.info("Loaded checkpoint from %s", ckpt_path)
    else:
        logger.warning(
            "Checkpoint not found at %s — using untrained weights for %s",
            ckpt_path, info["id"],
        )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# RTF benchmark (called by scripts/benchmark_rtf.py)
# ---------------------------------------------------------------------------

def benchmark_and_update_rtf(model_id: str, duration_s: float = 4.0) -> float:
    """Measure RTF and write it back into configs/models.yaml.

    Returns the measured RTF value.
    """
    engine = load_hybrid(model_id, force_reload=True)
    rtf = engine.rtf(duration_s=duration_s)

    # Update YAML in place
    with open(_REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    for entry in data["models"]:
        if entry["id"] == model_id:
            entry["rtf"] = round(rtf, 5)
    with open(_REGISTRY_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Updated RTF for %s = %.5f in %s", model_id, rtf, _REGISTRY_PATH)
    return rtf
