# YAMNet — Chatak Bioacoustics

**YAMNet TFLite model development, export and standalone C++ integration layer
for the Chatak wildlife acoustic-monitoring system.**

This repo is the *model* side of the Chatak stack.  The runtime firmware that
runs YAMNet inside ODAS lives in a separate repo (see [Related Repos](#related-repos)).

---

## Repository layout

```
.
├── integration/            ← C++ standalone classifier + integration tests
│   ├── export_yamnet_core.py  Build yamnet_core/ SavedModel + TFLite from TFHub
│   ├── yamnet_classifier.h/cpp  Reference C++ API (used as template for ODAS)
│   └── yamnet_custom/         yamnet.h5 saved model + tflite test
│
├── training/               ← YAMNet fine-tuning pipeline (NEW)
│   ├── data_loader.py         WAV + labels.csv → mel patches + tf.data
│   ├── train_yamnet.py        Two-phase fine-tuning (head freeze → top unfreeze)
│   └── export_finetuned.py    Export fine-tuned model → TFLite + registry
│
├── model_store/            ← Model store
│   ├── registry.json          Tracked: version history, class maps, accuracy
│   ├── base/                  (gitignored) yamnet_core/ SavedModel weights
│   ├── checkpoints/           (gitignored) training run outputs
│   └── releases/              (gitignored) versioned TFLite files for ODAS
│
└── models/ (submodule)     ← tensorflow/models (yamnet layer defs, read-only)
```

---

## Quick start — fine-tune on custom wildlife data

Dataset creation happens in the **simulator** repo (`yamnet_dataset_curator.py`).
Once you have a curator dataset, run:

```bash
# From ~/yamnet/, with tfenv activated
source tfenv/bin/activate

# 1. (First time) Export the base yamnet_core SavedModel
python integration/export_yamnet_core.py
# → writes  integration/yamnet_core/   (SavedModel)
# → writes  integration/yamnet_core.tflite

# Move base SavedModel to the model store
mkdir -p model_store/base
cp -r integration/yamnet_core  model_store/base/yamnet_core_savedmodel

# 2. Fine-tune
python training/train_yamnet.py \
    --dataset ~/simulator/outputs/yamnet_datasets/yamnet_train_001 \
    --savedmodel model_store/base/yamnet_core_savedmodel \
    --phase1-epochs 20 \
    --phase2-epochs 30
# → writes model_store/checkpoints/chatak_yamnet_<timestamp>/model.keras

# 3. Export to TFLite for ODAS
python training/export_finetuned.py \
    --checkpoint model_store/checkpoints/chatak_yamnet_<timestamp> \
    --version v1.0.0
# → writes model_store/releases/v1.0.0/chatak_yamnet_v1.0.0.tflite
# → writes model_store/releases/v1.0.0/custom_class_map.csv

# 4. Deploy to ODAS
cp model_store/releases/v1.0.0/chatak_yamnet_v1.0.0.tflite ~/sodas/
cp model_store/releases/v1.0.0/custom_class_map.csv        ~/sodas/
# Update raw.model_path and raw.class_map_path in your .cfg
```

---

## Quick start — export a fresh base model

```bash
# 1. Create Python environment
python3 -m venv tfenv && source tfenv/bin/activate
pip install tensorflow tensorflow-hub

# 2. Export YAMNet to TFLite
python3 integration/export_yamnet_core.py \
    --output integration/yamnet_core.tflite

# 3. Verify the model
cd integration
bash build.sh && bash test_pipeline.sh wavs/miaow_16k.wav
# Expected: class "Animal" or "Cat" with confidence > 0.5
```

---

## Quick start — build the standalone C++ classifier

```bash
cd integration

# Requires TFLite C lib (see SETUP.md for build instructions)
bash build_api_test.sh
bash run_api_test.sh wavs/miaow_16k.wav
```

See [integration/SETUP.md](integration/SETUP.md) for the full TFLite build guide.

---

## Model facts

| Property | Value |
|----------|-------|
| Architecture | YAMNet (MobileNet v1 backbone) |
| Input | 96 × 64 log-mel spectrogram patch |
| Output | 521 AudioSet class probabilities (sigmoid) |
| Sample rate | 16 000 Hz |
| Frame length | 400 samples (25 ms) |
| Frame hop | 160 samples (10 ms) |
| Spectrum bins required | **257** (512-pt FFT, do NOT truncate to 128) |
| Model size | ≈ 14 MB (float32 TFLite) |
| mAP (AudioSet eval) | 0.521 |

---

## Relationship to the ODAS firmware fork

The `integration/yamnet_classifier.h/.cpp` is the **reference implementation**
used as the starting point for the ODAS firmware.  The firmware carries its own
evolved copy in `src/yamnet/` which differs in these ways:

| Aspect | `~/yamnet/integration/` (this repo) | ODAS fork `src/yamnet/` |
|--------|--------------------------------------|-------------------------|
| TFLite API | C++ (`tflite::Interpreter`) | C (`TfLiteInterpreterCreate`) — better for embedded |
| TopK | ❌ single top-1 only | ✅ `ClassifyPatchTopK()` |
| Include path | `"yamnet_classifier.h"` | `"yamnet/yamnet_classifier.h"` |
| Model file | `integration/yamnet_core.tflite` | `models/yamnet_core.tflite` |

The two `.tflite` files are separately trained snapshots.
To sync: copy `integration/yamnet_core.tflite` into the ODAS fork's `models/`.

---

## Related repos

| Repo | Purpose |
|------|---------|
| [anamtya-tech/yamnet](https://github.com/anamtya-tech/yamnet) | **This repo** — model training, export, standalone C++ test |
| [anamtya-tech/simulator](https://github.com/anamtya-tech/simulator) | Python Streamlit pipeline — dataset curation, scene rendering, ODAS analysis |
| [anamtya-tech/chatak-odas](https://github.com/anamtya-tech/chatak-odas) | C fork of [introlab/odas](https://github.com/introlab/odas) with embedded YAMNet SST event pipeline |
| [DaveGamble/cJSON](https://github.com/DaveGamble/cJSON) | JSON library (submodule in ODAS fork) |
| [introlab/odas](https://github.com/introlab/odas) | Upstream ODAS (MIT) |

---

## License

- **YAMNet model weights**: Apache 2.0 ([Google](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet))
- **TensorFlow Lite**: Apache 2.0
- **Integration / wrapper code**: MIT
