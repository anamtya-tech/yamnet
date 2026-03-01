"""
Data loader for YAMNet fine-tuning.

Loads datasets in the format produced by yamnet_dataset_curator.py:

  dataset_dir/
    audio/       ← 16 kHz mono WAV files
    labels.csv   ← columns: filename, label, fold  (fold: train/val/test)
    metadata/    ← per-run provenance CSVs (ignored during training)

Converts WAV files to mel spectrogram patches using the EXACT same parameters
as the ODAS C++ YAMNet implementation (yamnet_classifier.cpp).  This ensures
that what we train on matches what ODAS infers at runtime.

Mel parameters (must stay in sync with yamnet_classifier.cpp / params.py):
  sample_rate      = 16000 Hz
  stft_window      = 400 samples  (25 ms)
  stft_hop         = 160 samples  (10 ms)
  fft_size         = 512
  spectrum_bins    = 257          (fft_size // 2 + 1)
  mel_bins         = 64
  mel_min_hz       = 125.0
  mel_max_hz       = 7500.0
  log_offset       = 0.001
  patch_frames     = 96           (~0.96 s of audio per patch)
  patch_hop        = 48           (50 % overlap)
"""

import csv
import sys
import os
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import tensorflow as tf

# ─── YAMNet mel parameters ────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
STFT_WINDOW    = 400       # samples
STFT_HOP       = 160       # samples
FFT_SIZE       = 512
MEL_BINS       = 64
MEL_MIN_HZ     = 125.0
MEL_MAX_HZ     = 7500.0
LOG_OFFSET     = 0.001
PATCH_FRAMES   = 96
PATCH_HOP      = 48


# ─── Mel filterbank (built once at import time) ───────────────────────────────
_MEL_FILTERBANK = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=MEL_BINS,
    num_spectrogram_bins=FFT_SIZE // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=MEL_MIN_HZ,
    upper_edge_hertz=MEL_MAX_HZ,
)  # shape (257, 64)


def waveform_to_mel_patches(waveform: np.ndarray) -> np.ndarray:
    """
    Convert a 16 kHz mono waveform to mel spectrogram patches.

    Args:
        waveform: float32 array, shape (n_samples,), values in [-1, 1].

    Returns:
        patches: float32 array, shape (N_patches, 96, 64, 1).
                 Returns empty array with shape (0, 96, 64, 1) if audio is
                 too short to form even a single patch.
    """
    wav_tensor = tf.cast(waveform, tf.float32)

    # STFT → magnitude
    stft = tf.signal.stft(
        wav_tensor,
        frame_length=STFT_WINDOW,
        frame_step=STFT_HOP,
        fft_length=FFT_SIZE,
        pad_end=True,
    )
    magnitude = tf.abs(stft)  # (n_frames, 257)

    # Mel filterbank
    mel = tf.tensordot(magnitude, _MEL_FILTERBANK, axes=[[1], [0]])  # (n_frames, 64)

    # Log compression
    log_mel = tf.math.log(mel + LOG_OFFSET)  # (n_frames, 64)

    # Frame into patches of PATCH_FRAMES with PATCH_HOP hop
    n_frames = tf.shape(log_mel)[0]

    if n_frames.numpy() < PATCH_FRAMES:
        return np.empty((0, PATCH_FRAMES, MEL_BINS, 1), dtype=np.float32)

    patches = tf.signal.frame(
        tf.transpose(log_mel),        # (64, n_frames) — frame along last axis
        frame_length=PATCH_FRAMES,
        frame_step=PATCH_HOP,
        axis=-1,
    )  # (64, n_patches, 96)

    patches = tf.transpose(patches, [1, 2, 0])  # (n_patches, 96, 64)
    patches = tf.expand_dims(patches, axis=-1)   # (n_patches, 96, 64, 1)

    return patches.numpy()


# ─── Label utilities ──────────────────────────────────────────────────────────

def build_label_map(labels_csv: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Scan labels.csv and build sorted class list and class→index mapping.

    Returns:
        classes:      sorted list of unique class names
        class_to_idx: dict mapping class name → integer index
    """
    raw_labels = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_labels.append(row["label"])
    classes = sorted(set(raw_labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_patches_for_split(
    dataset_dir: Path,
    fold: str,
    class_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all mel patches and labels for a given fold.

    Args:
        dataset_dir: Root of the curator dataset (contains audio/ and labels.csv).
        fold:        One of 'train', 'val', 'test'.
        class_to_idx: Mapping returned by build_label_map().

    Returns:
        X: float32 array, shape (N_patches, 96, 64, 1)
        y: int32 array,   shape (N_patches,)
    """
    labels_csv = dataset_dir / "labels.csv"
    audio_dir  = dataset_dir / "audio"

    X_list, y_list = [], []
    skipped = 0

    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("fold", "train") == fold]

    for row in rows:
        wav_path = audio_dir / row["filename"]
        label    = row["label"]

        if label not in class_to_idx:
            skipped += 1
            continue
        if not wav_path.exists():
            skipped += 1
            continue

        # Load WAV
        try:
            raw = tf.io.read_file(str(wav_path))
            waveform, sr = tf.audio.decode_wav(raw, desired_channels=1)
            waveform = tf.squeeze(waveform, axis=-1).numpy()  # (n,)
        except Exception as exc:
            print(f"  WARNING: could not load {wav_path}: {exc}")
            skipped += 1
            continue

        patches = waveform_to_mel_patches(waveform)
        if patches.shape[0] == 0:
            skipped += 1
            continue

        label_idx = class_to_idx[label]
        X_list.append(patches)
        y_list.extend([label_idx] * patches.shape[0])

    if not X_list:
        return (
            np.empty((0, PATCH_FRAMES, MEL_BINS, 1), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)

    if skipped:
        print(f"  fold={fold}: skipped {skipped} files (missing / too short)")

    return X, y


def load_dataset(
    dataset_dir: str,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Load train / val / test tf.data.Datasets from a curator dataset directory.

    If no val or test rows exist in labels.csv the function auto-splits:
      75 % train  /  15 % val  /  10 % test

    Args:
        dataset_dir: Path to curator dataset (contains audio/ and labels.csv).
        batch_size:  Batch size for all splits.
        seed:        Random seed for reproducible auto-split.

    Returns:
        train_ds, val_ds, test_ds: tf.data.Datasets of (patch, one_hot) pairs.
        classes:                   Sorted list of class names.
    """
    dataset_dir = Path(dataset_dir)
    labels_csv  = dataset_dir / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found in {dataset_dir}")

    classes, class_to_idx = build_label_map(labels_csv)
    num_classes = len(classes)
    print(f"\nClasses ({num_classes}): {classes}")

    # Detect whether fold column has val/test values
    with open(labels_csv, newline="", encoding="utf-8") as f:
        folds_present = {r.get("fold", "train") for r in csv.DictReader(f)}
    has_split = bool({"val", "test"} & folds_present)

    if has_split:
        X_tr, y_tr = load_patches_for_split(dataset_dir, "train", class_to_idx)
        X_va, y_va = load_patches_for_split(dataset_dir, "val",   class_to_idx)
        X_te, y_te = load_patches_for_split(dataset_dir, "test",  class_to_idx)
    else:
        # All rows are 'train' — auto-split
        X_all, y_all = load_patches_for_split(dataset_dir, "train", class_to_idx)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_all))
        n   = len(idx)
        n_v = int(n * 0.15)
        n_t = int(n * 0.10)

        X_va, y_va = X_all[idx[:n_v]],       y_all[idx[:n_v]]
        X_te, y_te = X_all[idx[n_v:n_v+n_t]], y_all[idx[n_v:n_v+n_t]]
        X_tr, y_tr = X_all[idx[n_v+n_t:]],   y_all[idx[n_v+n_t:]]
        print(f"Auto-split → train {len(X_tr)} / val {len(X_va)} / test {len(X_te)} patches")

    # Report class distribution
    for split_name, y_split, x_split in [("train", y_tr, X_tr),
                                          ("val",   y_va, X_va),
                                          ("test",  y_te, X_te)]:
        dist = Counter(classes[i] for i in y_split)
        print(f"  {split_name:5s}: {len(x_split):5d} patches  {dict(dist)}")

    def _make_ds(X, y, shuffle=False, augment=False):
        oh = tf.one_hot(y, num_classes)
        ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), oh))
        if shuffle:
            ds = ds.shuffle(len(X), reseed_each_iteration=True)
        if augment:
            def _aug(patch, label):
                noise = tf.random.normal(tf.shape(patch), stddev=0.02)
                gain  = tf.random.uniform([], 0.85, 1.15)
                return tf.clip_by_value(patch * gain + noise, -6.5, 4.0), label
            ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _make_ds(X_tr, y_tr, shuffle=True, augment=True)
    val_ds   = _make_ds(X_va, y_va)
    test_ds  = _make_ds(X_te, y_te)

    return train_ds, val_ds, test_ds, classes
