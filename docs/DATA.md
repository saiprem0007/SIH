# Data Preparation Guide

## Dataset Overview

| Dataset | Type | Size | Licence | Auto-download |
|---------|------|------|---------|---------------|
| LibriSpeech train-clean-100 | Speech | ~6 GB | CC BY 4.0 | Yes |
| LibriSpeech dev-clean | Speech | ~340 MB | CC BY 4.0 | Yes |
| LibriSpeech test-clean | Speech | ~346 MB | CC BY 4.0 | Yes |
| VCTK | Speech | ~10 GB | CC BY 4.0 | Manual |
| UrbanSound8K | Noise | ~6 GB | Custom (research) | Manual |
| ESC-50 | Noise | ~600 MB | CC BY-NC 3.0 | Yes |
| DEMAND | Noise | ~10 GB | Free for research | Manual |
| OpenSLR RIRs (SLR28) | Room IRs | ~1 GB | CC BY 4.0 | Yes |
| Freesound (manual CC0) | Noise | Variable | CC0 | Manual (CC0 only) |
| AudioSet (manual) | Noise | Variable | CC BY 4.0 (metadata) | Manual |

---

## 1. LibriSpeech

**Auto-download:**
```bash
python scripts/download_data.py --config configs/base.yaml --librispeech
```

Downloads `train-clean-100` (~6 GB), `dev-clean`, and `test-clean` via `torchaudio`.
Data is placed at `data/raw/LibriSpeech/`.

**Structure expected:**
```
data/raw/LibriSpeech/
  train-clean-100/
    {speaker_id}/{chapter_id}/*.flac
  dev-clean/...
  test-clean/...
```

---

## 2. VCTK

**Manual download** from: https://datashare.ed.ac.uk/handle/10283/3443

1. Download `VCTK-Corpus-0.92.zip`
2. Extract to `data/raw/VCTK-Corpus/`

**Structure expected:**
```
data/raw/VCTK-Corpus/
  wav48_silence_trimmed/
    p225/*.flac
    p226/*.flac
    ...
```

**Notes:** 110 speakers, British English accents (wide dialectal variety).

---

## 3. UrbanSound8K

**Manual download** from: https://urbansounddataset.weebly.com/urbansound8k.html

Requires free registration. Download `UrbanSound8K.tar.gz`, then:
```bash
tar -xzf UrbanSound8K.tar.gz -C data/raw/
```

**Structure expected:**
```
data/raw/UrbanSound8K/
  audio/
    fold1/*.wav   (classes: gun_shot, siren, engine_idling, jackhammer, ...)
    fold2/...
  metadata/UrbanSound8K.csv
```

**Key noise classes for defence scenarios:**
- `gun_shot` → category: impulsive
- `siren` → category: siren
- `engine_idling` → category: vehicle
- `jackhammer`, `drilling` → category: crowd

---

## 4. ESC-50

**Auto-download:**
```bash
python scripts/download_data.py --config configs/base.yaml --esc50
```

Downloads from GitHub and extracts to `data/raw/ESC-50-master/`.

**Licence:** CC BY-NC 3.0 — non-commercial use only.

---

## 5. DEMAND

**Manual download** from: https://zenodo.org/record/1227121

DEMAND provides recordings of 18 domestic and public environments at 48 kHz.
It cannot be programmatically downloaded.

1. Visit https://zenodo.org/record/1227121
2. Download the zip files for desired environments (e.g., NFIELD, OHALL, STRAFFIC)
3. Place in `data/raw/DEMAND/`
4. Resample to 16 kHz (optional — the pipeline handles this automatically):
   ```bash
   python -c "
   import sys; sys.path.insert(0,'src')
   from anc.data.download import resample_and_normalize_folder
   resample_and_normalize_folder('data/raw/DEMAND', 'data/processed/DEMAND', 16000)
   "
   ```

---

## 6. OpenSLR RIRs (SLR28)

**Auto-download:**
```bash
python scripts/download_data.py --config configs/base.yaml --openslr-rir
```

Downloads `rirs_noises.zip` (~1 GB) from OpenSLR and extracts to `data/raw/RIRS_NOISES/`.

Room impulse responses are used to simulate different room acoustics for speech.

---

## 7. Gunshot / Helicopter / Explosion Clips (Freesound — CC0 only)

> **Legal note:** Only CC0-licensed clips may be used for research with no attribution
> requirement. Do NOT use clips licensed CC BY-NC or proprietary.

**Recommended Freesound search queries (filter by CC0 licence):**
- `gunshot single`
- `helicopter rotor`
- `explosion outdoor`
- `artillery`
- `rocket launch`

**Download steps:**
1. Go to https://freesound.org
2. Search for each term, filter by **Licence: Creative Commons 0**
3. Download WAV/FLAC files
4. Place in `data/raw/freesound_manual/` (any subfolder structure)
5. Index them:
   ```bash
   python scripts/download_data.py --config configs/base.yaml \
       --index-manual data/raw/freesound_manual/
   ```

---

## 8. AudioSet (Optional)

AudioSet is a large-scale audio dataset from YouTube. The raw audio cannot be
scraped in bulk — each clip must be downloaded individually using `yt-dlp`.

**Prerequisites:**
```bash
pip install yt-dlp
```

**Download example** (download one 10-second segment):
```bash
yt-dlp -x --audio-format wav --audio-quality 0 \
    --postprocessor-args "-ar 16000 -ac 1" \
    -o "data/raw/audioset_manual/%(id)s.wav" \
    "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

**Licence:** Each YouTube video has its own licence. Only use CC-licensed videos.
AudioSet metadata files (class labels) are available at:
https://research.google.com/audioset/download.html

After downloading, index:
```bash
python scripts/download_data.py --config configs/base.yaml \
    --index-manual data/raw/audioset_manual/
```

---

## After Downloading All Data

```bash
# Build all manifests
python scripts/download_data.py --config configs/base.yaml --build-manifest

# Verify noise category distribution
python -c "
import pandas as pd
df = pd.read_csv('data/manifests/noise_train.csv')
print(df.groupby('category').size())
"

# If some categories are missing, the pipeline automatically generates
# synthetic substitutes (marked 'synthetic-cc0' in the manifest)
```

## Noise Category Distribution (with Real Data)

If some categories are missing from real datasets, the mixing pipeline
automatically generates synthetic substitutes:

| Category | Real data source | Synthetic fallback |
|----------|------------------|--------------------|
| impulsive | UrbanSound8K `gun_shot`, Freesound | `gen_synthetic_blast()` |
| rotor | Freesound / AudioSet | `gen_synthetic_rotor()` |
| vehicle | UrbanSound8K `engine_idling` | — |
| siren | UrbanSound8K `siren` | `gen_synthetic_siren()` |
| wind | ESC-50, Freesound | `gen_synthetic_wind()` |
| crowd | UrbanSound8K `jackhammer`, DEMAND | `gen_synthetic_crowd()` |
| misc | ESC-50 other, DEMAND | (all SYNTHETIC_GENERATORS) |

Synthetic noise is always included in the manifest to ensure all categories
have training examples even when real data is missing.
