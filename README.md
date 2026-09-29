# ANC-SIH: Real-Time Noise Cancellation for Defence Wearables

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Smart India Hackathon (SIH) prototype** — a real-time, causal, dual-microphone
speech enhancement system designed for helmet-mounted wearables in extreme defence
noise environments (gunshots, artillery, helicopter rotor, sirens, engine noise, wind).

---

## System Architecture

```
┌─────────────┐   ┌──────────────────┐   ┌─────────────────────┐   ┌──────────┐
│ Primary Mic │──▶│ Impulse Detector │──▶│   AI Enhancement    │──▶│          │
│(speech+noise│   │  (gunshot gate)  │   │ (causal TCN/DCU-Net)│   │  NLMS    │──▶ Output
└─────────────┘   └──────────────────┘   └─────────────────────┘   │  Filter  │
                                                                    │          │
┌─────────────┐                                                     │          │
│ Reference   │────────────────────────────────────────────────────▶│          │
│   Mic       │                                                     └──────────┘
│ (noise ref) │
└─────────────┘
```

The pipeline is **strictly causal**: no future look-ahead beyond the current frame.
Each stage carries state across frames for true streaming operation.

---

## Hardware Targets

| Target | SoC | Precision | Power |
|--------|-----|-----------|-------|
| **Primary** | Jetson Orin NX | FP16 / INT8 (TensorRT) | < 10 W |
| **Optional tiny** | Cortex-M / DSP | INT8 | < 1 W |

---

## Quality Targets

| Metric | Target |
|--------|--------|
| STOI (intelligibility) | > 0.85 |
| PESQ (wideband) | > 2.5 |
| Δ-SNR | ≈ 15 dB |
| End-to-end latency | < 40 ms |
| Frame / hop | 20 ms / 10 ms @ 16 kHz |

---

## Installation

```bash
# Clone and install in editable mode
cd anc/
pip install -e .

# Verify
python -c "import anc; print(anc.__version__)"
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.1, torchaudio, scipy, soundfile, pandas,
pystoi, pesq, librosa, matplotlib, pytest. See `requirements.txt`.

---

## Quick Start (≤ 10 commands)

```bash
# 1. Create demo dataset (no downloads needed — generates synthetic data)
python scripts/download_data.py --config configs/base.yaml --demo

# 2. Build manifests from available data
python scripts/download_data.py --config configs/base.yaml --build-manifest

# 3. Generate fixed, reproducible test set
python scripts/make_testset.py --config configs/base.yaml

# 4. Evaluate NLMS baseline
python scripts/evaluate.py --config configs/base.yaml --method nlms

# 5. Evaluate spectral-subtraction baseline
python scripts/evaluate.py --config configs/base.yaml --method spectral

# 6. Evaluate all methods + measure RTF
python scripts/evaluate.py --config configs/base.yaml --method all --measure-rtf

# 7. Run unit tests
pytest tests/ -v

# 8. Download real data (optional — large downloads)
python scripts/download_data.py --config configs/base.yaml --librispeech
python scripts/download_data.py --config configs/base.yaml --esc50
python scripts/download_data.py --config configs/base.yaml --openslr-rir

# 9. Re-build manifests with real data, regenerate test set
python scripts/download_data.py --config configs/base.yaml --build-manifest
python scripts/make_testset.py --config configs/base.yaml --force
```

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **0** | Data pipeline · baselines · eval harness | ✅ Complete |
| **1** | AI models (causal TCN · DCU-Net · Lite-GRU) + training | 🔲 Planned |
| **2** | Streaming hybrid pipeline + impulse detector | 🔲 Planned |
| **3** | Full evaluation · ablation study | 🔲 Planned |
| **4** | ONNX export · TensorRT INT8 · MCU deploy | 🔲 Planned |

---

## Repository Layout

```
anc/
├── configs/            YAML configuration files
├── data/               ← git-ignored: raw/, processed/, testset/, manifests/
├── src/anc/
│   ├── data/           Download, mixing, dataset, manifest, reference-mic
│   ├── baselines/      NLMS adaptive filter · Spectral subtraction
│   ├── models/         (Phase 1) TCN · DCU-Net · Lite-GRU
│   ├── losses/         (Phase 1) SI-SNR · waveform L1 · STFT loss
│   ├── pipeline/       (Phase 2) Streaming hybrid pipeline
│   ├── eval/           Metrics · evaluation runner
│   ├── deploy/         (Phase 4) ONNX export · TensorRT quantise
│   └── utils/          Config · seed · logging
├── scripts/            CLI entry points
├── tests/              Pytest unit tests
├── docs/               DATA.md · baseline_report.md
├── demo/               Demo notebooks / audio examples
└── runs/               Training logs (git-ignored)
```

---

## Assumptions

1. **Reference mic model** — Speech leakage simulated at −30 to −15 dB; real
   calibration deferred to Phase 4.
2. **Frame/hop** — 20 ms frame (320 samples @ 16 kHz), 10 ms hop (160 samples).
3. **Causality** — Model output for frame *t* depends only on frames ≤ *t* and
   carried hidden state. No non-causal normalisation.
4. **NLMS ordering** — Adaptive NLMS runs *after* the AI model, mopping up
   residual stationary leakage.
5. **Impulse detector** — Simple energy-ratio detector in Phase 0; upgraded to a
   learned detector in Phase 2.
6. **Dataset download** — Auto-downloads LibriSpeech, ESC-50, and OpenSLR RIRs.
   AudioSet/Freesound clips require manual CC0 download (see `docs/DATA.md`).
7. **Tiny model target** — ~50–100 k params, causal Conv1D + 1-layer GRU, INT8
   quantisable; MCU tooling deferred to Phase 4.
8. **OS** — Dev machine is Windows; all paths use `pathlib.Path` (cross-platform).
9. **Metrics availability** — STOI/PESQ computed only at eval time (require reference).
10. **ONNX/TRT** — Export deferred to Phase 4. Phase 0–3 targets PyTorch eager mode.

---

## Data

See [`docs/DATA.md`](docs/DATA.md) for download instructions and licence details.

---

## Contributing

This is an SIH hackathon project. Keep all changes config-driven, add unit tests
for new functionality, and update `PROGRESS.md` after each phase.
