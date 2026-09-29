"""
Evaluation metrics for speech enhancement.

Metrics computed:
  - SNR (input, output, delta)
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
  - STOI (Short-Time Objective Intelligibility)
  - PESQ (Perceptual Evaluation of Speech Quality, wideband)
  - RTF (Real-Time Factor)
"""
import logging
import time
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS: float = 1e-12
_PESQ_REQUIRED_SR: int = 16000  # wideband PESQ requires 16 kHz
_PESQ_MODE: str = "wb"

# Column labels for the markdown summary table
_MD_COLUMNS: List[str] = [
    "SNR Tier",
    "dSNR_mean",      # delta-SNR (ASCII safe for Windows console)
    "STOI_in",
    "STOI_out",
    "PESQ_in",
    "PESQ_out",
]


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_snr_db(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB between a clean reference and an estimate.

    SNR = 10 * log10( ||clean||^2 / ||clean - enhanced||^2 )

    Args:
        clean: Clean reference signal, 1-D float array.
        enhanced: Enhanced (or noisy) signal, 1-D float array, same length as
                  ``clean``.

    Returns:
        SNR in dB, or ``float('nan')`` if the clean signal is near-silent.
    """
    clean = np.asarray(clean, dtype=np.float64)
    enhanced = np.asarray(enhanced, dtype=np.float64)

    signal_power: float = float(np.sum(clean ** 2))
    if signal_power < _EPS:
        logger.warning("compute_snr_db: clean signal is near-silent; returning NaN.")
        return float("nan")

    noise_power: float = float(np.sum((clean - enhanced) ** 2)) + _EPS
    return float(10.0 * np.log10(signal_power / noise_power))


def compute_si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Both signals are zero-meaned before computation.

    SI-SDR = 10 * log10( ||alpha * s||^2 / ||alpha * s - e||^2 )
    where  alpha = <e, s> / <s, s>

    Args:
        ref: Clean reference signal, 1-D float array.
        est: Estimated (enhanced) signal, 1-D float array, same length.

    Returns:
        SI-SDR in dB, or ``float('nan')`` if the reference is zero.
    """
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)

    # Zero-mean
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    ref_energy: float = float(np.dot(ref, ref))
    if ref_energy < _EPS:
        logger.warning("compute_si_sdr: reference signal is zero; returning NaN.")
        return float("nan")

    # Optimal scaling factor
    alpha: float = float(np.dot(est, ref)) / ref_energy

    # Scaled reference and distortion
    s_alpha: np.ndarray = alpha * ref
    distortion: np.ndarray = s_alpha - est

    s_power: float = float(np.dot(s_alpha, s_alpha))
    d_power: float = float(np.dot(distortion, distortion)) + _EPS

    if s_power < _EPS:
        logger.warning("compute_si_sdr: scaled reference power is near-zero; returning NaN.")
        return float("nan")

    return float(10.0 * np.log10(s_power / d_power))


def compute_stoi(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Compute Short-Time Objective Intelligibility (STOI) score.

    Requires the ``pystoi`` package. Returns ``nan`` on any failure.

    Args:
        ref: Clean reference signal, 1-D array (any float dtype).
        est: Estimated signal, 1-D array (any float dtype).
        sr: Sampling rate in Hz.

    Returns:
        STOI score in ``[0, 1]`` (higher is more intelligible), or
        ``float('nan')`` on failure.
    """
    try:
        from pystoi import stoi  # type: ignore[import]
    except ImportError:
        logger.warning(
            "pystoi is not installed. Install it with: pip install pystoi. "
            "Returning NaN for STOI."
        )
        return float("nan")

    try:
        # pystoi requires float64
        ref_f64 = np.asarray(ref, dtype=np.float64)
        est_f64 = np.asarray(est, dtype=np.float64)

        # Align lengths to the shorter array
        min_len: int = min(ref_f64.shape[0], est_f64.shape[0])
        ref_f64 = ref_f64[:min_len]
        est_f64 = est_f64[:min_len]

        score: float = float(stoi(ref_f64, est_f64, sr, extended=False))
        return score
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_stoi failed: %s. Returning NaN.", exc)
        return float("nan")


def compute_pesq(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """Compute Perceptual Evaluation of Speech Quality (PESQ) — wideband mode.

    Requires the ``pesq`` package and a 16 kHz sampling rate.
    Returns ``nan`` on any failure.

    Args:
        ref: Clean reference signal, 1-D array (any float dtype).
        est: Estimated signal, 1-D array (any float dtype).
        sr: Sampling rate in Hz. Must be 16000 for wideband mode.

    Returns:
        PESQ MOS-LQO score (typically in ``[-0.5, 4.5]``), or
        ``float('nan')`` on failure.
    """
    if sr != _PESQ_REQUIRED_SR:
        logger.warning(
            "PESQ wideband requires sr=16000, got %d. Returning NaN.", sr
        )
        return float("nan")

    try:
        from pesq import pesq  # type: ignore[import]
    except ImportError:
        logger.warning(
            "pesq is not installed. Install it with: pip install pesq. "
            "Returning NaN for PESQ."
        )
        return float("nan")

    try:
        # pesq requires float32
        ref_f32 = np.asarray(ref, dtype=np.float32)
        est_f32 = np.asarray(est, dtype=np.float32)

        # Align lengths
        min_len: int = min(ref_f32.shape[0], est_f32.shape[0])
        ref_f32 = ref_f32[:min_len]
        est_f32 = est_f32[:min_len]

        score: float = float(pesq(sr, ref_f32, est_f32, _PESQ_MODE))
        return score
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_pesq failed: %s. Returning NaN.", exc)
        return float("nan")


def compute_all_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
    snr_tier: str = "",
    noise_category: str = "",
    sample_id: str = "",
) -> dict:
    """Compute a full suite of speech-enhancement metrics for one sample.

    Computes SNR, SI-SDR, STOI, and PESQ for both the noisy input and the
    enhanced output. All metric computations are wrapped individually so that
    one failure does not abort the rest.

    Args:
        clean: Ground-truth clean speech signal, 1-D float array.
        noisy: Noisy input signal (before enhancement), same length as ``clean``.
        enhanced: Enhanced output signal, same length as ``clean``.
        sr: Sampling rate in Hz.
        snr_tier: Human-readable SNR tier label (e.g. ``'0dB'``, ``'5dB'``).
        noise_category: Noise type label (e.g. ``'babble'``, ``'white'``).
        sample_id: Unique identifier for this sample.

    Returns:
        Dictionary with keys: ``sample_id``, ``snr_tier``, ``noise_category``,
        ``snr_in``, ``snr_out``, ``delta_snr``, ``si_sdr_in``, ``si_sdr_out``,
        ``si_sdr_delta``, ``stoi_in``, ``stoi_out``, ``pesq_in``, ``pesq_out``.

    Raises:
        ValueError: If arrays are not the same length or all are zero.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    enhanced = np.asarray(enhanced, dtype=np.float64)

    # Length validation
    min_len: int = min(clean.shape[0], noisy.shape[0], enhanced.shape[0])
    if min_len == 0:
        raise ValueError("Input arrays must not be empty.")

    if clean.shape[0] != noisy.shape[0] or clean.shape[0] != enhanced.shape[0]:
        logger.warning(
            "Array length mismatch (clean=%d, noisy=%d, enhanced=%d). "
            "Truncating to shortest.",
            clean.shape[0],
            noisy.shape[0],
            enhanced.shape[0],
        )
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        enhanced = enhanced[:min_len]

    # Zero-signal guard
    if np.max(np.abs(clean)) < _EPS:
        logger.warning(
            "sample_id='%s': clean signal appears to be all-zeros; "
            "metrics may be unreliable.",
            sample_id,
        )

    def _safe(fn: Callable, *args) -> float:  # noqa: ANN001
        """Call fn and return nan on any exception."""
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Metric computation failed (%s): %s", fn.__name__, exc)
            return float("nan")

    snr_in: float = _safe(compute_snr_db, clean, noisy)
    snr_out: float = _safe(compute_snr_db, clean, enhanced)
    delta_snr: float = (
        float("nan")
        if (np.isnan(snr_in) or np.isnan(snr_out))
        else snr_out - snr_in
    )

    si_sdr_in: float = _safe(compute_si_sdr, clean, noisy)
    si_sdr_out: float = _safe(compute_si_sdr, clean, enhanced)
    si_sdr_delta: float = (
        float("nan")
        if (np.isnan(si_sdr_in) or np.isnan(si_sdr_out))
        else si_sdr_out - si_sdr_in
    )

    stoi_in: float = _safe(compute_stoi, clean, noisy, sr)
    stoi_out: float = _safe(compute_stoi, clean, enhanced, sr)

    pesq_in: float = _safe(compute_pesq, clean, noisy, sr)
    pesq_out: float = _safe(compute_pesq, clean, enhanced, sr)

    return {
        "sample_id": sample_id,
        "snr_tier": snr_tier,
        "noise_category": noise_category,
        "snr_in": snr_in,
        "snr_out": snr_out,
        "delta_snr": delta_snr,
        "si_sdr_in": si_sdr_in,
        "si_sdr_out": si_sdr_out,
        "si_sdr_delta": si_sdr_delta,
        "stoi_in": stoi_in,
        "stoi_out": stoi_out,
        "pesq_in": pesq_in,
        "pesq_out": pesq_out,
    }


def aggregate_results(
    results: List[dict],
    group_by: str = "snr_tier",
) -> pd.DataFrame:
    """Aggregate a list of per-sample metric dicts into group-level statistics.

    Args:
        results: List of dicts as returned by :func:`compute_all_metrics`.
        group_by: Column name to group by. Typically ``'snr_tier'`` or
                  ``'noise_category'``.

    Returns:
        DataFrame with ``group_by`` as the index. For each numeric metric,
        columns ``<metric>_mean`` and ``<metric>_std`` are added.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results list is empty; nothing to aggregate.")

    df: pd.DataFrame = pd.DataFrame(results)

    if group_by not in df.columns:
        raise ValueError(
            f"group_by column '{group_by}' not found in results. "
            f"Available columns: {list(df.columns)}"
        )

    numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()

    agg_dict = {col: ["mean", "std"] for col in numeric_cols}
    grouped: pd.DataFrame = df.groupby(group_by).agg(agg_dict)

    # Flatten MultiIndex columns
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    grouped.index.name = group_by

    return grouped


def results_to_markdown(
    df: pd.DataFrame,
    title: str = "Evaluation Results",
) -> str:
    """Format an aggregated results DataFrame as a markdown table.

    Only includes the summary columns relevant for human-readable reporting:
    ΔSNR mean, STOI in/out, PESQ in/out.

    Args:
        df: DataFrame as returned by :func:`aggregate_results` (index =
            group_by, columns include ``*_mean`` / ``*_std`` suffixes).
        title: Markdown heading to place above the table.

    Returns:
        Markdown string with a level-2 heading and a GFM-compatible table.
    """
    # Select and rename columns for the summary table
    col_map: dict = {
        "delta_snr_mean": "dSNR_mean",    # ASCII-safe (was DELTA-SNR_mean)
        "stoi_in_mean": "STOI_in",
        "stoi_out_mean": "STOI_out",
        "pesq_in_mean": "PESQ_in",
        "pesq_out_mean": "PESQ_out",
    }

    # Only include columns that actually exist in the dataframe
    available_cols: List[str] = [c for c in col_map if c in df.columns]
    summary: pd.DataFrame = df[available_cols].copy().rename(columns=col_map)
    summary = summary.round(3)
    summary.index.name = "SNR Tier"

    lines: List[str] = [f"## {title}", ""]

    # Build header row
    headers: List[str] = [summary.index.name] + list(summary.columns)
    header_row: str = "| " + " | ".join(headers) + " |"
    separator_row: str = "| " + " | ".join(["---"] * len(headers)) + " |"

    lines.append(header_row)
    lines.append(separator_row)

    for idx, row in summary.iterrows():
        values: List[str] = [str(idx)] + [
            f"{v:.3f}" if not (isinstance(v, float) and np.isnan(v)) else "n/a"
            for v in row.values
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def results_to_csv(
    results: List[dict],
    path: Union[str, Path],
) -> None:
    """Save a flat list of per-sample metric dicts to a CSV file.

    The parent directory is created automatically if it does not exist.

    Args:
        results: List of metric dicts as returned by :func:`compute_all_metrics`.
        path: Destination file path (str or :class:`~pathlib.Path`).

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results list is empty; nothing to save.")

    out_path: Path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d metric rows to '%s'.", len(results), out_path)


def plot_results(
    results: List[dict],
    output_dir: Union[str, Path],
    method_name: str = "method",
) -> None:
    """Generate and save evaluation bar charts.

    Creates two PNG files in ``output_dir``:

    * ``delta_snr_per_tier_<method_name>.png`` — ΔSNR grouped by SNR tier.
    * ``stoi_per_noise_<method_name>.png`` — STOI_out grouped by noise category.

    Args:
        results: List of metric dicts as returned by :func:`compute_all_metrics`.
        output_dir: Directory where PNG files will be saved.
        method_name: Label used in file names and plot titles.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_results. "
            "Install it with: pip install matplotlib"
        ) from exc

    if not results:
        logger.warning("plot_results: results list is empty; skipping plots.")
        return

    out_dir: Path = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame = pd.DataFrame(results)

    # ---- Plot 1: ΔSNR per SNR tier ----------------------------------------
    if "snr_tier" in df.columns and "delta_snr" in df.columns:
        tier_grouped = (
            df.dropna(subset=["delta_snr"])
            .groupby("snr_tier")["delta_snr"]
            .agg(["mean", "std"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = range(len(tier_grouped))
        ax.bar(
            x_pos,
            tier_grouped["mean"],
            yerr=tier_grouped["std"].fillna(0.0),
            capsize=4,
            color="steelblue",
            alpha=0.85,
            label=method_name,
        )
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(tier_grouped["snr_tier"].tolist(), rotation=30, ha="right")
        ax.set_xlabel("SNR Tier")
        ax.set_ylabel("ΔSNR (dB)")
        ax.set_title(f"ΔSNR per SNR Tier — {method_name}")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend()
        fig.tight_layout()

        plot_path_1: Path = out_dir / f"delta_snr_per_tier_{method_name}.png"
        fig.savefig(plot_path_1, dpi=150)
        plt.close(fig)
        logger.info("Saved plot: %s", plot_path_1)
    else:
        logger.warning(
            "plot_results: 'snr_tier' or 'delta_snr' column missing; "
            "skipping ΔSNR plot."
        )

    # ---- Plot 2: STOI_out per noise category --------------------------------
    if "noise_category" in df.columns and "stoi_out" in df.columns:
        noise_grouped = (
            df.dropna(subset=["stoi_out"])
            .groupby("noise_category")["stoi_out"]
            .agg(["mean", "std"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = range(len(noise_grouped))
        ax.bar(
            x_pos,
            noise_grouped["mean"],
            yerr=noise_grouped["std"].fillna(0.0),
            capsize=4,
            color="darkorange",
            alpha=0.85,
            label=method_name,
        )
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(noise_grouped["noise_category"].tolist(), rotation=30, ha="right")
        ax.set_xlabel("Noise Category")
        ax.set_ylabel("STOI_out")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"STOI_out per Noise Category — {method_name}")
        ax.legend()
        fig.tight_layout()

        plot_path_2: Path = out_dir / f"stoi_per_noise_{method_name}.png"
        fig.savefig(plot_path_2, dpi=150)
        plt.close(fig)
        logger.info("Saved plot: %s", plot_path_2)
    else:
        logger.warning(
            "plot_results: 'noise_category' or 'stoi_out' column missing; "
            "skipping STOI plot."
        )


def measure_rtf(
    process_fn: Callable[[np.ndarray], np.ndarray],
    audio_dur_s: float = 4.0,
    sr: int = _PESQ_REQUIRED_SR,
    n_repeats: int = 5,
) -> float:
    """Measure the Real-Time Factor (RTF) of a processing function.

    RTF = (mean wall-clock processing time) / (audio duration)

    A value < 1.0 means the system runs faster than real-time.

    Args:
        process_fn: A callable that accepts a 1-D ``np.float32`` array of audio
                    samples and returns a 1-D array of the same length.
        audio_dur_s: Duration of the test audio clip in seconds.
        sr: Sampling rate in Hz used to synthesise the test signal.
        n_repeats: Number of repeated calls for averaging.

    Returns:
        Mean RTF across ``n_repeats`` runs.
    """
    n_samples: int = int(audio_dur_s * sr)
    test_audio: np.ndarray = np.random.randn(n_samples).astype(np.float32)

    elapsed_times: List[float] = []
    for i in range(n_repeats):
        t_start: float = time.perf_counter()
        _ = process_fn(test_audio.copy())
        t_end: float = time.perf_counter()
        elapsed_times.append(t_end - t_start)

    mean_elapsed: float = float(np.mean(elapsed_times))
    rtf: float = mean_elapsed / audio_dur_s

    logger.info(
        "RTF measurement: audio_dur=%.2fs, n_repeats=%d, "
        "mean_processing_time=%.4fs, RTF=%.4f",
        audio_dur_s,
        n_repeats,
        mean_elapsed,
        rtf,
    )
    return rtf
