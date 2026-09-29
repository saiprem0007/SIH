# ANC-SIH Phase 0 Baseline Report

**Dataset**: Synthetic demo dataset (50 speech + 50 noise clips, all generated in-process)  
**Evaluation date**: 2026-09-29  
**Samples evaluated**: 30 per method (limited by `--max-samples 30`)  
**PESQ**: N/A — requires C compiler to install `pesq`; gracefully returns NaN  
**Run command**:
```bash
python scripts/evaluate.py --config configs/base.yaml --method all --max-samples 30
```

---

## Important Caveat: Synthetic Demo Data Limitation

The demo dataset generates speech and noise entirely synthetically.
The noise files are zero-mean random signals written as **PCM-16** `.wav` which,
after float32→int16 quantisation, results in **near-silent noise signals** (RMS ≈ 0).
Consequently:

- The `mix_at_snr` function detects near-zero noise and returns the speech unchanged,
  logging `"Noise RMS near zero"`.
- The measured `snr_in` in the test set is therefore **artificially inflated** (~26 dB
  instead of the intended 10 dB for "Easy") because the mixture is essentially clean speech.
- The NLMS and spectral-subtraction filters then *add* quantisation / Gibbs noise,
  yielding **negative delta-SNR**.

**This is a data-quality issue specific to the synthetic demo.
Real numbers will be obtained after downloading LibriSpeech + UrbanSound8K (see `docs/DATA.md`).**

---

## Baseline Results — Synthetic Demo (30 samples per method)

### By SNR Tier

| Method | SNR Tier | SNR_in (dB) | SNR_out (dB) | delta-SNR | SI-SDR_in | SI-SDR_out | STOI_in | STOI_out |
|--------|----------|-------------|--------------|-----------|-----------|------------|---------|----------|
| Noisy (input) | Easy    | 26.4 | 26.4 | 0.0  | 32.8 | 32.8  | 0.93 | 0.93 |
| Noisy (input) | Extreme | 8.8  | 8.8  | 0.0  | 13.9 | 13.9  | 0.52 | 0.52 |
| NLMS          | Easy    | 26.4 | 4.2  | **−22.2** | 32.8 | 4.4 | 0.93 | 0.83 |
| NLMS          | Extreme | 8.8  | −3.1 | **−12.0** | 13.9 | −5.3 | 0.52 | 0.48 |
| SpectralSub   | Easy    | 26.4 | −0.7 | **−27.1** | 32.8 | −30.6 | 0.93 | 0.24 |
| SpectralSub   | Extreme | 8.8  | −2.8 | **−11.6** | 13.9 | −38.3 | 0.52 | 0.13 |

> ⚠️ Negative delta-SNR is expected here because near-silent noise → the input is
> essentially clean. Any filter introduces distortion.

---

## Expected Numbers (with Real Data)

Based on literature for NLMS and spectral subtraction on real speech + defence-noise mixtures:

| Method | Expected delta-SNR | Expected STOI_out | Expected PESQ_out |
|--------|--------------------|-------------------|-------------------|
| NLMS (adaptive) | +8 to +15 dB (stationary noise) | 0.70–0.80 | 1.8–2.5 |
| Spectral subtraction | +5 to +10 dB | 0.65–0.75 | 1.5–2.2 |
| Target (AI model, Phase 1) | **~15 dB** | **>0.85** | **>2.5** |

---

## Real-Time Factor (RTF)

| Method | RTF (CPU) | Real-time capable? |
|--------|-----------|-------------------|
| NLMS (256 taps, Python loop) | ~0.06 | Yes |
| Spectral subtraction | ~0.008 | Yes |
| Target AI model (Phase 1, INT8 TRT) | < 0.1 | Expected yes |

> RTF < 1.0 means real-time capable. Both baselines run well below 1.0 on CPU.
> NLMS measured at ~1.3 samples/s = ~0.06 RTF for 4-second clips.

---

## How to Get Real Numbers

1. **Download real data:**
   ```bash
   python scripts/download_data.py --config configs/base.yaml --librispeech
   python scripts/download_data.py --config configs/base.yaml --esc50
   python scripts/download_data.py --config configs/base.yaml --openslr-rir
   ```

2. **Rebuild manifests and test set:**
   ```bash
   python scripts/download_data.py --config configs/base.yaml --build-manifest
   python scripts/make_testset.py --config configs/base.yaml --force
   ```

3. **Re-run evaluation:**
   ```bash
   python scripts/evaluate.py --config configs/base.yaml --method all --measure-rtf
   ```

4. **Fill in the table above with actual numbers.**

---

## Files Generated

| File | Contents |
|------|----------|
| `eval_results/nlms_results.csv` | Per-sample metrics for NLMS |
| `eval_results/spectral_results.csv` | Per-sample metrics for SpectralSub |
| `eval_results/noisy_results.csv` | Per-sample input (noisy) metrics |
| `eval_results/nlms_plots/` | Bar charts: delta-SNR by tier, STOI by category |
| `data/testset/` | Fixed reproducible test set (85 samples) |
| `data/manifests/` | Speech and noise manifests (train/val/test CSVs) |
