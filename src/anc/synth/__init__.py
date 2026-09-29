"""
ANC-SIH Phase 4 -- Synthetic Defence Noise Generators.

All generators return float32 numpy arrays at 16 kHz (or any sr),
peak-normalised to ~0.90.  All randomness flows through a caller-supplied
np.random.Generator for full reproducibility.

Quick usage::

    from anc.synth import blast, gunshot, DEFENCE_GENERATORS, render_all
    rng = np.random.default_rng(42)
    wav = blast(4.0, 16000, rng)
    render_all("assets/samples/gallery", duration_s=4.0, sr=16000)
"""
from anc.synth.generators import (
    blast,
    gunshot,
    artillery,
    rotor,
    siren,
    engine,
    tracked_vehicle,
    wind_buffet,
    babble,
    DEFENCE_GENERATORS,
    render_all,
)

__all__ = [
    "blast",
    "gunshot",
    "artillery",
    "rotor",
    "siren",
    "engine",
    "tracked_vehicle",
    "wind_buffet",
    "babble",
    "DEFENCE_GENERATORS",
    "render_all",
]
