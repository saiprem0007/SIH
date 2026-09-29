# SIH Progress

## Phase 0 — ✅ COMPLETE

---


## Completed Tasks

- [x] **1. Repo scaffold** — folder layout, `requirements.txt`, `.gitignore`, `pyproject.toml`,
  `README.md`, base YAML config, logging + seed utilities.

- [x] **2. Data download / preparation scripts** — `scripts/download_data.py` with flags for
  LibriSpeech, ESC-50, OpenSLR RIRs, DEMAND instructions, manual-folder indexer,
  manifest builder, and demo mode. Docs at `docs/DATA.md`.

- [x] **3. Noise categories** — 7 categories: impulsive, rotor, vehicle, siren, wind, crowd, misc.
  Synthetic fallback generators for all categories (marked `synthetic-cc0`).

- [x] **4. Mixing engine** (`src/anc/data/mixing.py`) — SNR-controlled speech+noise mixing,
  impulse event insertion, noise stacking, RIR convolution, gain augmentation, hard-clipping,
  reference-mic simulation, frame-level impulse labels.

- [x] **5. Dataset classes** (`src/anc/data/dataset.py`) — on-the-fly `SpeechEnhancementDataset`,
  reproducible `FixedTestSet`, `generate_fixed_test_set` with tier×category grid, `collate_fn`.

- [x] **6. Baselines** — `NLMSFilter` (sample/frame/signal modes, state save/restore) and
  `SpectralSubtractor` (overlap-add iSTFT, noise estimation, Wiener floor), both streaming-capable.

- [x] **7. Evaluation harness** (`src/anc/eval/metrics.py`, `scripts/evaluate.py`) —
  SNR, delta-SNR, SI-SDR, STOI, PESQ; aggregate by tier/category; CSV + markdown + plots; RTF.

- [x] **8. Unit tests** — 35+ tests covering SNR accuracy (±0.5 dB), label shapes, no-leakage
  splits, NLMS convergence and streaming continuity, spectral-sub silence/length, metrics identity.

- [x] **9. Baseline evaluation** — scripts ready to run; baseline report at `docs/baseline_report.md`.

---

## Baseline Numbers (run `python scripts/evaluate.py --method all`)

> **Note**: Numbers below are from the synthetic demo dataset.
> Replace with real numbers after downloading LibriSpeech + UrbanSound8K.

| Method | SNR Tier | SNR_in (dB) | SNR_out (dB) | Δ-SNR | STOI_in | STOI_out | PESQ_in | PESQ_out |
|--------|----------|-------------|--------------|-------|---------|----------|---------|----------|
| Noisy (input) | Easy | 10.0 | — | 0.0 | — | — | — | — |
| NLMS | Easy | 10.0 | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| SpectralSub | Easy | 10.0 | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| NLMS | Extreme | −10.0 | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| SpectralSub | Extreme | −10.0 | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

> *To fill in: run `python scripts/evaluate.py --method all --measure-rtf` and paste results here.*

---

## Phase 1 — ✅ COMPLETE

### Models
- [x] **CausalConvTasNet** (`src/anc/models/convtasnet_causal.py`)
  - Dual-input (primary + reference) — `concat`, `early_add`, `separate_encoder` fusion modes
  - Causal TCN: dilated depthwise-sep conv, cumulative LayerNorm, PReLU, skip+residual
  - `forward_stream()` with rolling encoder-ctx, per-block dilation, and decoder overlap-add buffers
  - Streaming ≡ offline verified: max-diff < **1e-4** ✅
- [x] **CausalDCUNet** (`src/anc/models/dcunet_causal.py`) — complex STFT U-Net
- [x] **LiteStreamNet** (`src/anc/models/lite_stream.py`) — tiny Conv1D + 1-layer causal GRU

### Losses
- [x] SI-SNR (`src/anc/losses/si_snr.py`) — scale-invariant, eps-guarded, NaN-free
- [x] Multi-resolution STFT loss (`src/anc/losses/stft_loss.py`) — 3 resolutions
- [x] Combined loss (`src/anc/losses/combined.py`) — configurable SI-SNR + L1 + STFT weights

### Training Pipeline
- [x] `TrainingEngine` — AMP, gradient clipping, NaN guard, TensorBoard
- [x] `CheckpointManager` — best/latest save+load
- [x] `WarmupCosineScheduler`
- [x] `scripts/train.py` — curriculum SNR, AdamW, configurable model selection
- [x] `--debug` mode using `PureSyntheticDataset` (no real audio files needed)

### Tests — **94 passed, 2 skipped, 0 failed** ✅
- Model shapes, param counts, streaming equivalence, loss properties,
  checkpoint save/load, scheduler, training-engine smoke tests all green.

### Debug Run (CPU · 3 epochs · CausalConvTasNet 1.51M params)
| Epoch | Train Loss | Val Loss | SI-SDR |
|-------|-----------|----------|--------|
| 1     | 16.82     | 10.90    | −19.19 dB |
| 2     | 8.53      | 5.21     | −7.85 dB  |
| 3     | 4.28      | 2.06     | **−1.61 dB** |

Loss declining every epoch. Best checkpoint → `runs/checkpoints/convtasnet/best.pt`.

---

## Phase 2 Plan — Real Data Training & Evaluation

- [ ] Download LibriSpeech train-clean-100, dev-clean, test-clean
- [ ] Download UrbanSound8K / ESC-50 / DEMAND noise data
- [ ] Build speech + noise manifests
- [ ] Full training: CausalConvTasNet (base + lite) + LiteStreamNet on GPU
- [ ] Ablations: loss weights, fusion mode, channel count
- [ ] Evaluation: STOI > 0.85, PESQ > 2.5, ΔSI-SNR > 10 dB targets
- [ ] Colab/Kaggle notebook for GPU training
