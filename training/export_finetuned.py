"""
Export a fine-tuned YAMNet checkpoint to TFLite for ODAS C++ deployment.

The exported TFLite model:
  - Input  : (1, 96, 64, 1) float32 — mel spectrogram patch
             (identical format to the base yamnet_core.tflite used by ODAS)
  - Output : (1, N)           float32 — softmax probabilities over N custom classes

The ODAS C++ classifier (yamnet_classifier.cpp) needs only two changes:
  1. Point model_path to the new .tflite file.
  2. Point class_map_path to the new custom_class_map.csv.

Usage
-----
  python training/export_finetuned.py \\
      --checkpoint  model_store/checkpoints/chatak_yamnet_20260301_120000 \
      [--version    v1.0.0] \
      [--output-dir model_store/releases]

Outputs
-------
  model_store/releases/<version>/
      chatak_yamnet_<version>.tflite        ← float32 (for ODAS)
      chatak_yamnet_<version>_int8.tflite   ← INT8 quantized (smaller / faster)
      custom_class_map.csv                  ← index,class_name
      export_info.json                      ← provenance metadata
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent


def load_checkpoint(checkpoint_dir: Path) -> tuple[tf.keras.Model, list[str]]:
    """
    Load a fine-tuned model and its class map from a training checkpoint dir.

    Returns:
        model:   Loaded Keras model.
        classes: List of class names in label-index order.
    """
    model_path     = checkpoint_dir / "model.keras"
    class_map_path = checkpoint_dir / "class_map.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"model.keras not found in {checkpoint_dir}")
    if not class_map_path.exists():
        raise FileNotFoundError(f"class_map.csv not found in {checkpoint_dir}")

    print(f"Loading model from {model_path} …")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print(f"  ✓ Model loaded  — output shape: {model.output_shape}")

    classes = []
    with open(class_map_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if parts[0] == "index":
                continue
            classes.append(parts[1])
    print(f"  ✓ Classes ({len(classes)}): {classes}")

    return model, classes


def export_tflite(
    model: tf.keras.Model,
    output_path: Path,
    quantize: bool = False,
) -> bool:
    """
    Convert Keras model to TFLite and write to output_path.

    Args:
        model:       Fine-tuned Keras model.
        output_path: Destination .tflite path.
        quantize:    Apply INT8 dynamic-range quantisation if True.

    Returns:
        True on success.
    """
    print(f"\n{'INT8 quantized' if quantize else 'Float32'} TFLite export → {output_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"  ✓ Written {size_mb:.2f} MB")

    # Sanity-check inference
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    dummy = np.random.randn(*inp["shape"]).astype(inp["dtype"])
    interp.set_tensor(inp["index"], dummy)
    interp.invoke()
    preds = interp.get_tensor(out["index"])

    print(f"  Input  shape : {inp['shape']}  dtype: {inp['dtype'].__name__}")
    print(f"  Output shape : {out['shape']}  dtype: {out['dtype'].__name__}")
    print(f"  Test sum : {preds.sum():.4f}  (should be ≈ 1.0 for softmax)")

    if not (0.99 < preds.sum() < 1.01):
        print("  ⚠  Probabilities do not sum to 1.0 — check model output activation")

    return True


def write_class_map(classes: list[str], output_path: Path) -> None:
    """Write a CSV class map compatible with the ODAS C++ loader."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("index,display_name\n")
        for i, c in enumerate(classes):
            f.write(f"{i},{c}\n")
    print(f"  Class map → {output_path}")


def _update_registry(
    registry_path: Path,
    run_name: str,
    version: str,
    tflite_path: Path,
    tflite_int8_path: Path,
    classes: list[str],
    timestamp: str,
) -> None:
    if not registry_path.exists():
        reg = {"schema_version": "1", "models": [], "active_model": None}
    else:
        with open(registry_path) as f:
            reg = json.load(f)

    for entry in reg["models"]:
        if entry.get("run_name") == run_name:
            entry["tflite_path"]      = str(tflite_path)
            entry["tflite_int8_path"] = str(tflite_int8_path)
            entry["version"]          = version
            entry["exported_at"]      = timestamp
            break
    else:
        # run_name not in registry yet — create minimal entry
        reg["models"].append({
            "run_name":        run_name,
            "version":         version,
            "timestamp":       timestamp,
            "exported_at":     timestamp,
            "classes":         classes,
            "num_classes":     len(classes),
            "tflite_path":     str(tflite_path),
            "tflite_int8_path": str(tflite_int8_path),
            "deployed":        False,
        })

    with open(registry_path, "w") as f:
        json.dump(reg, f, indent=2)
    print(f"  Registry    → {registry_path}")


def export(
    checkpoint_dir: str,
    version:        str = "v1.0.0",
    output_dir:     str = "model_store/releases",
) -> Path:
    """
    Full export pipeline.

    Returns the release directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    release_dir    = REPO_ROOT / output_dir / version
    release_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = checkpoint_dir.name

    print("\n" + "=" * 70)
    print(f"YAMNET EXPORT  —  version: {version}  run: {run_name}")
    print("=" * 70)

    # Load checkpoint
    model, classes = load_checkpoint(checkpoint_dir)

    # Paths
    tflite_path      = release_dir / f"chatak_yamnet_{version}.tflite"
    tflite_int8_path = release_dir / f"chatak_yamnet_{version}_int8.tflite"
    class_map_path   = release_dir / "custom_class_map.csv"
    info_path        = release_dir / "export_info.json"

    # Export float32 TFLite
    export_tflite(model, tflite_path, quantize=False)

    # Export INT8 quantized TFLite
    export_tflite(model, tflite_int8_path, quantize=True)

    # Write class map
    write_class_map(classes, class_map_path)

    # Write provenance JSON
    info = {
        "version":         version,
        "exported_at":     ts,
        "run_name":        run_name,
        "checkpoint_dir":  str(checkpoint_dir),
        "classes":         classes,
        "num_classes":     len(classes),
        "tflite_float32":  str(tflite_path),
        "tflite_int8":     str(tflite_int8_path),
        "class_map":       str(class_map_path),
        "odas_integration": {
            "note": "Copy tflite and class_map to ~/sodas/ or the ODAS build dir.",
            "config_keys": {
                "raw.model_path":      str(tflite_path),
                "raw.class_map_path":  str(class_map_path),
            },
        },
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Update registry
    _update_registry(
        registry_path    = REPO_ROOT / "model_store" / "registry.json",
        run_name         = run_name,
        version          = version,
        tflite_path      = tflite_path,
        tflite_int8_path = tflite_int8_path,
        classes          = classes,
        timestamp        = ts,
    )

    print("\n" + "=" * 70)
    print(f"Export complete → {release_dir}")
    print("\nTo deploy to ODAS:")
    print(f"  cp {tflite_path} ~/sodas/")
    print(f"  cp {class_map_path} ~/sodas/")
    print(f"  # Update raw.model_path and raw.class_map_path in your .cfg")
    print("=" * 70)

    return release_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Export a fine-tuned YAMNet checkpoint to TFLite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to training checkpoint directory")
    p.add_argument("--version",    default="v1.0.0",
                   help="Semantic version string for this release")
    p.add_argument("--output-dir", default="model_store/releases",
                   help="Parent directory for versioned releases")
    return p.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(REPO_ROOT / "integration"))

    args = _parse_args()
    export(
        checkpoint_dir = args.checkpoint,
        version        = args.version,
        output_dir     = args.output_dir,
    )
