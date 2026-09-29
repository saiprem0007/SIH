"""
Synthetic defence-noise generators for ANC-SIH Phase 4.

Each function signature::

    fn(duration_s: float, sr: int, rng: np.random.Generator, **kwargs) -> np.ndarray

Returns float32, peak-normalised to PEAK_TARGET (~0.90).
All filtering uses scipy.signal; no librosa dependency.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

# ---------------------------------------------------------------------------
# Physical-model constants (match params.yaml exactly)
# ---------------------------------------------------------------------------

PEAK_TARGET: float = 0.90

# blast
BLAST_RISE_MS_MIN:       float = 0.5
BLAST_RISE_MS_MAX:       float = 2.0
BLAST_DECAY_MIN:         float = 20.0
BLAST_DECAY_MAX:         float = 80.0
BLAST_THUMP_F:           float = 60.0
BLAST_THUMP_DB:          float = -6.0
BLAST_ONSET_MAX_FRAC:    float = 0.25
BLAST_N_ECHOES_MAX:      int   = 3
BLAST_ECHO_DELAY_MAX_MS: float = 80.0
BLAST_ECHO_DECAY_DB:     float = -12.0

# gunshot
GUN_RISE_MS:           float = 0.3
GUN_TAIL_DECAY_MIN:    float = 5.0
GUN_TAIL_DECAY_MAX:    float = 30.0
GUN_HF_LOSS_MIN:       float = 1000.0
GUN_HF_LOSS_MAX:       float = 6000.0
GUN_ECHO_COUNT_MAX:    int   = 5

# artillery
ART_F_DOMINANT:        float = 40.0
ART_DECAY_MIN:         float = 2.0
ART_DECAY_MAX:         float = 8.0
ART_ECHO_COUNT:        int   = 6
ART_ECHO_SPREAD_MAX_MS:float = 300.0
ART_LP_CUTOFF:         float = 300.0

# rotor
ROTOR_F0_MIN:          float = 15.0
ROTOR_F0_MAX:          float = 30.0
ROTOR_N_HARMONICS:     int   = 12
ROTOR_AM_RATE_MIN:     float = 0.3
ROTOR_AM_RATE_MAX:     float = 1.5
ROTOR_AM_DEPTH_MIN:    float = 0.05
ROTOR_AM_DEPTH_MAX:    float = 0.20
ROTOR_HISS_DB:         float = -15.0

# siren
SIREN_F_LO:            float = 300.0
SIREN_F_HI:            float = 3000.0
SIREN_SWEEP_MIN:       float = 0.5
SIREN_SWEEP_MAX:       float = 2.0
SIREN_N_HARMONICS:     int   = 4
SIREN_AM_DEPTH:        float = 0.10
SIREN_WAIL_PROB:       float = 0.5

# engine
ENGINE_RPM_MIN:        int   = 600
ENGINE_RPM_MAX:        int   = 3000
ENGINE_N_HARMONICS:    int   = 8
ENGINE_BROADBAND_DB:   float = -10.0
ENGINE_RPM_SWEEP_PROB: float = 0.40

# tracked_vehicle
TRACKED_RUMBLE_F_LO:      float = 20.0
TRACKED_RUMBLE_F_HI:      float = 200.0
TRACKED_CLANK_RATE_MIN:   float = 2.0
TRACKED_CLANK_RATE_MAX:   float = 15.0
TRACKED_CLANK_DUR_MS_MIN: float = 2.0
TRACKED_CLANK_DUR_MS_MAX: float = 10.0
TRACKED_CLANK_DB_MIN:     float = -6.0
TRACKED_CLANK_DB_MAX:     float = 6.0

# wind_buffet
WIND_BUFFET_F_LP:      float = 400.0
WIND_GUST_RATE_MIN:    float = 0.2
WIND_GUST_RATE_MAX:    float = 2.0
WIND_GUST_DEPTH_MIN:   float = 0.3
WIND_GUST_DEPTH_MAX:   float = 0.8

# babble
BABBLE_N_SPEAKERS_MIN:              int   = 4
BABBLE_N_SPEAKERS_MAX:              int   = 8
BABBLE_PITCH_SHIFT_SEMITONES_MAX:   float = 3.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _peak_norm(sig: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    peak = float(np.max(np.abs(sig)))
    if peak < 1e-9:
        return sig.astype(np.float32)
    return (sig * (target / peak)).astype(np.float32)


def _lp(fc: float, sr: int, order: int = 4) -> Tuple:
    nyq = sr / 2.0
    wn = float(np.clip(fc / nyq, 1e-4, 1.0 - 1e-4))
    return scipy_signal.butter(order, wn, btype="low")


def _bp(lo: float, hi: float, sr: int, order: int = 4) -> Tuple:
    nyq = sr / 2.0
    low  = float(np.clip(lo / nyq, 1e-4, 1.0 - 1e-4))
    high = float(np.clip(hi / nyq, 1e-4, 1.0 - 1e-4))
    if low >= high:
        high = min(low + 0.01, 1.0 - 1e-4)
    return scipy_signal.butter(order, [low, high], btype="band")


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise via spectral shaping."""
    white = rng.standard_normal(n)
    fft_v = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = np.inf
    fft_v *= 1.0 / np.sqrt(freqs)
    fft_v[0] = 0.0
    return np.fft.irfft(fft_v, n=n).astype(np.float64)


def _add_echoes(
    base: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    n_echoes: int,
    max_delay_ms: float,
    decay_db_per_echo: float,
) -> np.ndarray:
    """Add time-delayed, attenuated echo copies to *base*."""
    out = base.copy()
    for i in range(1, n_echoes + 1):
        delay_s = rng.uniform(0.005, max_delay_ms / 1000.0)
        delay_n = int(delay_s * sr)
        gain    = 10.0 ** (decay_db_per_echo * i / 20.0)
        if delay_n < len(base):
            out[delay_n:] += gain * base[: len(base) - delay_n]
    return out


# ---------------------------------------------------------------------------
# 1. blast
# ---------------------------------------------------------------------------

def blast(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    n_echoes: Optional[int] = None,
) -> np.ndarray:
    """Sharp-onset broadband burst with low-frequency thump and optional reflections.

    Physical model:
    * Noise burst with fast linear-ramp attack (< 2 ms) and exponential decay.
    * Low-frequency thump (60 Hz sine) added at BLAST_THUMP_DB.
    * 0..BLAST_N_ECHOES_MAX delayed reflections, each attenuated by BLAST_ECHO_DECAY_DB.
    """
    N = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, N, endpoint=False, dtype=np.float64)

    onset_idx = int(rng.uniform(0.0, BLAST_ONSET_MAX_FRAC) * N)

    # Attack envelope
    rise_n = max(1, int(rng.uniform(BLAST_RISE_MS_MIN, BLAST_RISE_MS_MAX) * sr / 1000.0))
    decay_rate = rng.uniform(BLAST_DECAY_MIN, BLAST_DECAY_MAX)
    t_rel = np.maximum(0.0, t - t[onset_idx])
    attack = np.minimum(t_rel / (rise_n / sr), 1.0)
    envelope = attack * np.exp(-decay_rate * t_rel)

    burst = rng.standard_normal(N) * envelope

    # Low-frequency thump
    thump_gain = 10.0 ** (BLAST_THUMP_DB / 20.0)
    thump_decay = decay_rate * 0.3   # thump decays slower
    thump = thump_gain * np.sin(2.0 * np.pi * BLAST_THUMP_F * t) * attack * np.exp(-thump_decay * t_rel)
    burst += thump

    # Echoes
    if n_echoes is None:
        n_echoes = int(rng.integers(0, BLAST_N_ECHOES_MAX + 1))
    if n_echoes > 0:
        burst = _add_echoes(burst, sr, rng, n_echoes, BLAST_ECHO_DELAY_MAX_MS, BLAST_ECHO_DECAY_DB)

    return _peak_norm(burst)


# ---------------------------------------------------------------------------
# 2. gunshot
# ---------------------------------------------------------------------------

def gunshot(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    distance_m: Optional[float] = None,
) -> np.ndarray:
    """N-wave ballistic transient with reverberant tail and distance modelling.

    Physical model:
    * Bipolar N-wave pulse (muzzle blast shape): positive lobe then negative.
    * Exponentially decaying reverberant tail (broadband).
    * Distance: low-pass filtering (air absorption) + level attenuation.
    * Room echoes: GUN_ECHO_COUNT_MAX reflections.
    """
    N = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, N, endpoint=False, dtype=np.float64)

    # N-wave: bipolar pulse
    rise_n = max(2, int(GUN_RISE_MS * sr / 1000.0))
    onset  = int(0.05 * N)          # onset at 5%
    sig    = np.zeros(N, dtype=np.float64)
    # Positive lobe
    pos_end = min(onset + rise_n, N)
    sig[onset:pos_end] = np.linspace(0.0, 1.0, pos_end - onset)
    # Negative lobe
    neg_end = min(pos_end + rise_n, N)
    sig[pos_end:neg_end] = np.linspace(-1.0, 0.0, neg_end - pos_end)

    # Reverberant tail
    tail_decay = rng.uniform(GUN_TAIL_DECAY_MIN, GUN_TAIL_DECAY_MAX)
    t_tail = np.maximum(0.0, t - t[min(neg_end, N - 1)])
    tail = rng.standard_normal(N) * np.exp(-tail_decay * t_tail) * 0.3
    sig += tail

    # Distance modelling
    if distance_m is None:
        distance_m = float(rng.uniform(1.0, 200.0))
    level_db = -20.0 * np.log10(max(distance_m, 1.0))
    sig *= 10.0 ** (level_db / 20.0)

    fc_base = rng.uniform(GUN_HF_LOSS_MIN, GUN_HF_LOSS_MAX)
    fc = max(200.0, fc_base / max(distance_m, 1.0))
    b, a = _lp(min(fc, sr * 0.45), sr)
    sig = scipy_signal.lfilter(b, a, sig)

    # Echoes
    n_echoes = int(rng.integers(1, GUN_ECHO_COUNT_MAX + 1))
    sig = _add_echoes(sig, sr, rng, n_echoes, max_delay_ms=200.0, decay_db_per_echo=-6.0)

    return _peak_norm(sig)


# ---------------------------------------------------------------------------
# 3. artillery
# ---------------------------------------------------------------------------

def artillery(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Low-frequency dominant artillery blast with rolling echo tail.

    Physical model:
    * Dominant fundamental at ~40 Hz + 2nd/3rd harmonic (low-frequency boom).
    * Slow exponential decay envelope.
    * Low-pass filtered at 300 Hz (mostly subsonic / low rumble).
    * Rolling echo tail: ART_ECHO_COUNT reflections spread up to ART_ECHO_SPREAD_MAX_MS.
    """
    N = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, N, endpoint=False, dtype=np.float64)

    decay_rate = rng.uniform(ART_DECAY_MIN, ART_DECAY_MAX)
    onset_idx  = int(0.05 * N)
    t_rel = np.maximum(0.0, t - t[onset_idx])
    envelope = np.exp(-decay_rate * t_rel)
    envelope[:onset_idx] = 0.0

    # Harmonic boom
    phases = rng.uniform(0.0, 2.0 * np.pi, 3)
    boom  = np.sin(2.0 * np.pi * ART_F_DOMINANT       * t + phases[0])
    boom += 0.5 * np.sin(2.0 * np.pi * ART_F_DOMINANT * 2 * t + phases[1])
    boom += 0.25 * np.sin(2.0 * np.pi * ART_F_DOMINANT * 3 * t + phases[2])
    boom *= envelope

    # Broadband component (low-passed)
    broadband = rng.standard_normal(N) * envelope * 0.4
    b, a = _lp(ART_LP_CUTOFF, sr)
    broadband = scipy_signal.lfilter(b, a, broadband)

    sig = boom + broadband

    # Rolling echo tail
    sig = _add_echoes(sig, sr, rng, ART_ECHO_COUNT,
                      max_delay_ms=ART_ECHO_SPREAD_MAX_MS, decay_db_per_echo=-4.0)

    return _peak_norm(sig)


# ---------------------------------------------------------------------------
# 4. rotor
# ---------------------------------------------------------------------------

def rotor(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    f0: Optional[float] = None,
) -> np.ndarray:
    """Helicopter / drone rotor noise with harmonic series, AM, and turbine hiss.

    Physical model:
    * Blade-pass fundamental f0 (15-30 Hz) + ROTOR_N_HARMONICS harmonics (1/k amplitude).
    * Amplitude modulation at 0.3-1.5 Hz with random depth.
    * Broadband turbine hiss at ROTOR_HISS_DB below the fundamental.
    """
    N  = int(duration_s * sr)
    t  = np.linspace(0.0, duration_s, N, endpoint=False, dtype=np.float64)

    if f0 is None:
        f0 = float(rng.uniform(ROTOR_F0_MIN, ROTOR_F0_MAX))

    phases = rng.uniform(0.0, 2.0 * np.pi, ROTOR_N_HARMONICS)
    sig    = np.zeros(N, dtype=np.float64)
    for k in range(1, ROTOR_N_HARMONICS + 1):
        freq = k * f0
        if freq >= sr / 2.0:
            break
        sig += (1.0 / k) * np.sin(2.0 * np.pi * freq * t + phases[k - 1])

    # Amplitude modulation
    am_rate  = rng.uniform(ROTOR_AM_RATE_MIN, ROTOR_AM_RATE_MAX)
    am_depth = rng.uniform(ROTOR_AM_DEPTH_MIN, ROTOR_AM_DEPTH_MAX)
    am_phase = rng.uniform(0.0, 2.0 * np.pi)
    am = 1.0 - am_depth * np.sin(2.0 * np.pi * am_rate * t + am_phase)
    sig *= am

    # Turbine hiss
    fund_rms = float(np.sqrt(np.mean(sig ** 2))) + 1e-10
    hiss_gain = fund_rms * 10.0 ** (ROTOR_HISS_DB / 20.0)
    sig += rng.standard_normal(N) * hiss_gain

    return _peak_norm(sig)


# ---------------------------------------------------------------------------
# 5. siren
# ---------------------------------------------------------------------------

def siren(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    pattern: Optional[str] = None,
) -> np.ndarray:
    """Emergency siren with wail or yelp FM sweep pattern and harmonics.

    Physical model:
    * Wail: slow sinusoidal FM sweep (0.5-2 s period).
    * Yelp: fast repetitive sweep (0.1-0.3 s period).
    * SIREN_N_HARMONICS harmonics with 1/k amplitude.
    * Mild AM wobble at SIREN_AM_DEPTH.
    """
    N = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, N, endpoint=False, dtype=np.float64)

    if pattern is None:
        pattern = "wail" if rng.random() < SIREN_WAIL_PROB else "yelp"

    if pattern == "wail":
        sweep_period = rng.uniform(SIREN_SWEEP_MIN, SIREN_SWEEP_MAX)
    else:
        sweep_period = rng.uniform(0.1, 0.3)

    f_mid = (SIREN_F_LO + SIREN_F_HI) / 2.0
    f_dev = (SIREN_F_HI - SIREN_F_LO) / 2.0
    inst_freq = f_mid + f_dev * np.sin(2.0 * np.pi * t / sweep_period)

    sig = np.zeros(N, dtype=np.float64)
    for k in range(1, SIREN_N_HARMONICS + 1):
        phase = 2.0 * np.pi * np.cumsum(inst_freq * k) / sr
        sig += (1.0 / k) * np.sin(phase)

    # AM wobble
    am = 1.0 - SIREN_AM_DEPTH * np.sin(2.0 * np.pi * t / sweep_period)
    sig *= am

    return _peak_norm(sig)


# ---------------------------------------------------------------------------
# 6. engine
# ---------------------------------------------------------------------------

def engine(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    rpm: Optional[int] = None,
) -> np.ndarray:
    """Internal combustion engine: harmonic series from RPM + broadband.

    Physical model:
    * Fundamental at rpm/60 Hz + ENGINE_N_HARMONICS harmonics (1/k amplitude).
    * Optional RPM sweep (idle->rev up or rev down).
    * Broadband noise floor at ENGINE_BROADBAND_DB below fundamental.
    """
    N = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, N, endpoint=False, dtype=np.float64)

    if rpm is None:
        rpm = int(rng.integers(ENGINE_RPM_MIN, ENGINE_RPM_MAX))

    # Optionally sweep RPM
    if rng.random() < ENGINE_RPM_SWEEP_PROB:
        rpm_start = int(rng.integers(ENGINE_RPM_MIN, ENGINE_RPM_MAX))
        rpm_end   = int(rng.integers(ENGINE_RPM_MIN, ENGINE_RPM_MAX))
        rpm_arr   = np.linspace(rpm_start, rpm_end, N, dtype=np.float64)
    else:
        rpm_arr = np.full(N, float(rpm), dtype=np.float64)

    f0_arr = rpm_arr / 60.0    # instantaneous fundamental frequency

    phases = rng.uniform(0.0, 2.0 * np.pi, ENGINE_N_HARMONICS)
    sig = np.zeros(N, dtype=np.float64)
    for k in range(1, ENGINE_N_HARMONICS + 1):
        inst_freq = k * f0_arr
        inst_freq = np.clip(inst_freq, 0.0, sr / 2.0 - 1.0)
        phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr + phases[k - 1]
        sig += (1.0 / k) * np.sin(phase)

    # Broadband noise
    fund_rms   = float(np.sqrt(np.mean(sig ** 2))) + 1e-10
    broad_gain = fund_rms * 10.0 ** (ENGINE_BROADBAND_DB / 20.0)
    sig       += rng.standard_normal(N) * broad_gain

    return _peak_norm(sig)


# ---------------------------------------------------------------------------
# 7. tracked_vehicle
# ---------------------------------------------------------------------------

def tracked_vehicle(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tracked military vehicle: low rumble + irregular metallic clanks.

    Physical model:
    * Rumble: bandpass-filtered noise (20-200 Hz) — drivetrain vibration.
    * Clanks: Poisson-distributed broadband impulses (track link impacts).
    """
    N = int(duration_s * sr)

    # Rumble
    white = rng.standard_normal(N)
    b, a  = _bp(TRACKED_RUMBLE_F_LO, TRACKED_RUMBLE_F_HI, sr)
    rumble = scipy_signal.lfilter(b, a, white)
    rumble_rms = float(np.sqrt(np.mean(rumble ** 2))) + 1e-10

    # Clanks (Poisson process)
    clank_rate = rng.uniform(TRACKED_CLANK_RATE_MIN, TRACKED_CLANK_RATE_MAX)
    n_clanks   = rng.poisson(clank_rate * duration_s)
    sig = rumble.copy()

    for _ in range(n_clanks):
        dur_ms  = rng.uniform(TRACKED_CLANK_DUR_MS_MIN, TRACKED_CLANK_DUR_MS_MAX)
        c_len   = max(2, int(dur_ms * sr / 1000.0))
        db_rel  = rng.uniform(TRACKED_CLANK_DB_MIN, TRACKED_CLANK_DB_MAX)
        c_gain  = rumble_rms * 10.0 ** (db_rel / 20.0)
        onset   = int(rng.integers(0, max(1, N - c_len)))
        clank   = rng.standard_normal(c_len) * c_gain
        clank  *= np.hanning(c_len)
        sig[onset: onset + c_len] += clank

    return _peak_norm(sig)


# ---------------------------------------------------------------------------
# 8. wind_buffet
# ---------------------------------------------------------------------------

def wind_buffet(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Microphone wind buffeting: low-passed gusty noise with random envelope.

    Physical model:
    * White noise low-passed at 400 Hz (aerodynamic pressure fluctuations).
    * Baseline envelope 0.4, Poisson-distributed gust events modulate amplitude.
    """
    N = int(duration_s * sr)

    # Base low-passed noise
    white = rng.standard_normal(N)
    b, a  = _lp(WIND_BUFFET_F_LP, sr)
    base  = scipy_signal.lfilter(b, a, white).astype(np.float64)

    # Gust envelope
    gust_rate = rng.uniform(WIND_GUST_RATE_MIN, WIND_GUST_RATE_MAX)
    n_gusts   = rng.poisson(gust_rate * duration_s)
    envelope  = np.full(N, 0.4, dtype=np.float64)

    for _ in range(n_gusts):
        gust_dur = rng.uniform(0.3, 1.5)          # gust duration seconds
        g_len    = min(int(gust_dur * sr), N)
        onset    = int(rng.integers(0, max(1, N - g_len)))
        depth    = rng.uniform(WIND_GUST_DEPTH_MIN, WIND_GUST_DEPTH_MAX)
        gust_env = depth * np.hanning(g_len)
        envelope[onset: onset + g_len] += gust_env

    return _peak_norm(base * envelope)


# ---------------------------------------------------------------------------
# 9. babble
# ---------------------------------------------------------------------------

def babble(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    speech_files: Optional[List[str]] = None,
) -> np.ndarray:
    """Cocktail-party babble: sum of N overlapping speech streams.

    If speech_files provided: loads N random clips, offsets them randomly,
    optionally pitch-shifts via resampling trick.
    Fallback (no files): bandpass-filtered noise (200-4000 Hz).
    """
    N = int(duration_s * sr)
    n_speakers = int(rng.integers(BABBLE_N_SPEAKERS_MIN, BABBLE_N_SPEAKERS_MAX + 1))

    if speech_files:
        try:
            import soundfile as sf
            from math import gcd

            sig = np.zeros(N, dtype=np.float64)
            for _ in range(n_speakers):
                fpath = speech_files[int(rng.integers(0, len(speech_files)))]
                audio, file_sr = sf.read(str(fpath), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                # Resample if needed
                if file_sr != sr:
                    g = gcd(file_sr, sr)
                    audio = scipy_signal.resample_poly(audio, sr // g, file_sr // g)

                # Optional pitch shift via resampling (proxy)
                semitones = rng.uniform(
                    -BABBLE_PITCH_SHIFT_SEMITONES_MAX, BABBLE_PITCH_SHIFT_SEMITONES_MAX
                )
                ratio = 2.0 ** (semitones / 12.0)
                if abs(semitones) > 0.1:
                    stretched_len = int(len(audio) / ratio)
                    if stretched_len > 0:
                        audio = scipy_signal.resample(audio, stretched_len)

                # Random time offset and crop/pad to N
                if len(audio) >= N:
                    onset = int(rng.integers(0, len(audio) - N + 1))
                    chunk = audio[onset: onset + N]
                else:
                    chunk = np.tile(audio, int(np.ceil(N / len(audio))))[:N]

                # Random level
                gain = rng.uniform(0.5, 1.5)
                sig += gain * chunk.astype(np.float64)

            return _peak_norm(sig)
        except Exception as exc:
            warnings.warn(f"babble: error loading speech files ({exc}); using noise fallback.")

    # Fallback: bandpass noise proxy
    white = rng.standard_normal(N)
    b, a  = _bp(200.0, 4000.0, sr)
    proxy = scipy_signal.lfilter(b, a, white)
    return _peak_norm(proxy)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DEFENCE_GENERATORS: Dict[str, Callable] = {
    "blast":           blast,
    "gunshot":         gunshot,
    "artillery":       artillery,
    "rotor":           rotor,
    "siren":           siren,
    "engine":          engine,
    "tracked_vehicle": tracked_vehicle,
    "wind_buffet":     wind_buffet,
    "babble":          babble,
}
"""All 9 defence-noise generators, keyed by name."""


# ---------------------------------------------------------------------------
# render_all
# ---------------------------------------------------------------------------

def render_all(
    output_dir,
    duration_s: float = 4.0,
    sr: int = 16000,
    seed: int = 42,
) -> Dict[str, Path]:
    """Render one WAV clip per generator and save to *output_dir*.

    Args:
        output_dir: Target directory (created if missing).
        duration_s: Clip duration in seconds.
        sr:         Sample rate in Hz.
        seed:       RNG seed.

    Returns:
        Dict mapping generator name -> Path of saved WAV.
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for render_all") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: Dict[str, Path] = {}

    for name, fn in DEFENCE_GENERATORS.items():
        wav = fn(duration_s, sr, rng)
        fpath = out / f"{name}.wav"
        sf.write(str(fpath), wav, sr)
        paths[name] = fpath

    return paths
