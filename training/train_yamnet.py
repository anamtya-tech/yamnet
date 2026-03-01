"""
Fine-tune YAMNet on a custom wildlife classification dataset.

Architecture
------------
  Input  : (batch, 96, 64, 1)  — mel spectrogram patch
           (same format as the current yamnet_core.tflite used by ODAS)
  Backbone: YAMNet depthwise-separable CNN → GlobalAveragePooling → (batch, 1024)
            Weights loaded from yamnet_core/ SavedModel or TFHub.
            Frozen during Phase 1; top layers unfrozen in Phase 2.
  Head   : Dense(256, relu) → Dropout(0.3) → Dense(N, softmax)

Two-phase training
------------------
  Phase 1 – head only (backbone frozen):
      optimizer = Adam(lr=1e-3), epochs = phase1_epochs (default 20)
  Phase 2 – optional fine-tune top backbone layers:
      optimizer = Adam(lr=1e-5), epochs = phase2_epochs (default 30)

Usage
-----
  python training/train_yamnet.py \\
      --dataset  /home/azureuser/simulator/outputs/yamnet_datasets/yamnet_train_001 \\
      --savedmodel  integration/yamnet_core \\
      [--hub-url  https://tfhub.dev/google/yamnet/1] \\
      [--phase1-epochs 20] \\
      [--phase2-epochs 30] \\
      [--batch-size 32] \\
      [--output-dir  model_store/checkpoints]

The script writes:
  model_store/checkpoints/<run_name>/
      model.keras          ← full fine-tuned Keras model
      class_map.csv        ← index,class_name
      training_log.json    ← metrics, hyperparams, dataset path
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

# ── Resolve repo root (so the script works from any cwd) ────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_ROOT    = SCRIPT_DIR.parent
MODELS_DIR   = REPO_ROOT / "model_store" / "checkpoints"

# Add models submodule to path for YAMNet layer definitions
YAMNET_MODELS_PATH = REPO_ROOT / "models" / "research" / "audioset" / "yamnet"
if YAMNET_MODELS_PATH.exists():
    sys.path.insert(0, str(YAMNET_MODELS_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Model building
# ─────────────────────────────────────────────────────────────────────────────

def _build_yamnet_layers(params):
    """Import and return the YAMNet layer definitions from the models submodule."""
    from yamnet import _YAMNET_LAYER_DEFS  # type: ignore[import]
    return _YAMNET_LAYER_DEFS


def build_finetuned_model(
    num_classes: int,
    savedmodel_path: str | None = None,
    hub_url: str | None = None,
) -> tf.keras.Model:
    """
    Build the fine-tuned YAMNet model.

    Weight source priority:
      1. savedmodel_path  (local SavedModel produced by export_yamnet_core.py)
      2. hub_url          (TFHub — requires internet)
      3. Random init      (training from scratch — not recommended)

    Args:
        num_classes:     Number of custom output classes.
        savedmodel_path: Path to yamnet_core/ SavedModel directory.
        hub_url:         TFHub URL fallback.

    Returns:
        Keras Model with backbone + custom head.
        Backbone layers are initially frozen (trainable=False).
    """
    import params as params_module  # type: ignore[import]  # from models submodule

    params = params_module.Params()
    layer_defs = _build_yamnet_layers(params)

    # ── Build backbone ───────────────────────────────────────────────────────
    mel_input  = tf.keras.Input(shape=(96, 64, 1), name="mel_spectrogram")
    net = mel_input
    for i, (layer_fun, kernel, stride, filters) in enumerate(layer_defs):
        net = layer_fun(f"layer{i + 1}", kernel, stride, filters, params)(net)

    embeddings = tf.keras.layers.GlobalAveragePooling2D(name="embeddings")(net)

    # ── Custom head ──────────────────────────────────────────────────────────
    x = tf.keras.layers.Dense(256, activation="relu", name="head_fc")(embeddings)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)
    predictions = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="custom_predictions"
    )(x)

    model = tf.keras.Model(inputs=mel_input, outputs=predictions, name="yamnet_finetuned")

    # ── Load pre-trained backbone weights ────────────────────────────────────
    weights_loaded = False

    if savedmodel_path and Path(savedmodel_path).exists():
        print(f"\nLoading backbone weights from SavedModel: {savedmodel_path}")
        weights_loaded = _load_from_savedmodel(model, savedmodel_path, params)

    if not weights_loaded and hub_url:
        print(f"\nLoading backbone weights from TFHub: {hub_url}")
        weights_loaded = _load_from_hub(model, hub_url, params)

    if not weights_loaded:
        print("\n⚠  No pre-trained weights loaded — training from random init.")
        print("   Pass --savedmodel or --hub-url for transfer learning.")

    # ── Freeze backbone (all layers except the 3 head layers) ────────────────
    head_names = {"head_fc", "head_dropout", "custom_predictions"}
    for layer in model.layers:
        layer.trainable = layer.name in head_names

    trainable = sum(1 for l in model.layers if l.trainable)
    total     = len(model.layers)
    print(f"\nFrozen backbone: {total - trainable} layers frozen, "
          f"{trainable} head layers trainable")

    return model


def _load_from_savedmodel(
    model: tf.keras.Model,
    savedmodel_path: str,
    params,
) -> bool:
    """Transfer backbone weights from a local yamnet_core SavedModel."""
    try:
        import params as params_module  # type: ignore[import]
        from export_yamnet_core import yamnet_core_model  # type: ignore[import]

        # Build reference model (521-class) and load SavedModel weights
        ref_model = yamnet_core_model(params)
        ref_model.load_weights(savedmodel_path)

        # Copy all backbone variables (everything before the new head)
        head_names = {"head_fc/kernel", "head_fc/bias",
                      "custom_predictions/kernel", "custom_predictions/bias"}
        src_vars = {v.name: v for v in ref_model.variables}
        copied = 0
        for var in model.variables:
            # Strip the model-name prefix for matching
            short = "/".join(var.name.split("/")[1:])
            if short in head_names:
                continue
            if short in src_vars and src_vars[short].shape == var.shape:
                var.assign(src_vars[short])
                copied += 1

        print(f"  ✓ Copied {copied} backbone variables from SavedModel")
        return copied > 0
    except Exception as exc:
        print(f"  ✗ SavedModel load failed: {exc}")
        return False


def _load_from_hub(model: tf.keras.Model, hub_url: str, params) -> bool:
    """Transfer backbone weights from a TFHub YAMNet model."""
    try:
        import tensorflow_hub as hub  # type: ignore[import]
        from export_yamnet_core import yamnet_core_model, transfer_all_weights  # type: ignore[import]

        yamnet_hub = hub.load(hub_url)
        ref_model  = yamnet_core_model(params)

        # Build ref model with a dummy forward pass
        _ = ref_model(tf.zeros((1, 96, 64, 1)))
        transferred = transfer_all_weights(yamnet_hub, ref_model)
        if transferred == 0:
            return False

        # Copy backbone weights from ref_model → finetuned model
        src_vars  = {v.name: v for v in ref_model.variables}
        head_keys = {"head_fc", "custom_predictions"}
        copied = 0
        for var in model.variables:
            if any(k in var.name for k in head_keys):
                continue
            short = "/".join(var.name.split("/")[1:])
            if short in src_vars and src_vars[short].shape == var.shape:
                var.assign(src_vars[short])
                copied += 1

        print(f"  ✓ Copied {copied} backbone variables from TFHub")
        return copied > 0
    except Exception as exc:
        print(f"  ✗ TFHub load failed: {exc}")
        return False


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int = 4) -> None:
    """
    Unfreeze the top N backbone layers for Phase 2 fine-tuning.

    Skips batch-norm layers to keep running statistics stable.
    """
    backbone_layers = [
        l for l in model.layers
        if l.name not in {"head_fc", "head_dropout", "custom_predictions"}
    ]
    to_unfreeze = backbone_layers[-n_layers:]
    for layer in to_unfreeze:
        if "batch_norm" not in layer.name.lower():
            layer.trainable = True
            print(f"  Unfreezing: {layer.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    dataset_dir:    str,
    savedmodel_path: str | None = None,
    hub_url:         str | None = None,
    phase1_epochs:  int   = 20,
    phase2_epochs:  int   = 30,
    batch_size:     int   = 32,
    output_dir:     str   = "model_store/checkpoints",
    unfreeze_top:   int   = 4,
    run_name:       str | None = None,
) -> Path:
    """
    Full two-phase training pipeline.

    Returns the path to the output checkpoint directory.
    """
    # Import here so the module is usable without a full TF install
    from data_loader import load_dataset  # relative import within training/

    # ── Timestamp run ────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"chatak_yamnet_{ts}"
    ckpt_dir = Path(output_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"YAMNET FINE-TUNING  —  run: {run_name}")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, classes = load_dataset(
        dataset_dir, batch_size=batch_size
    )
    num_classes = len(classes)
    if num_classes < 2:
        raise ValueError(f"Need ≥ 2 classes, got {num_classes}: {classes}")

    # ── Save class map ───────────────────────────────────────────────────────
    class_map_path = ckpt_dir / "class_map.csv"
    with open(class_map_path, "w") as f:
        f.write("index,class_name\n")
        for i, c in enumerate(classes):
            f.write(f"{i},{c}\n")
    print(f"\nClass map saved → {class_map_path}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_finetuned_model(
        num_classes=num_classes,
        savedmodel_path=savedmodel_path,
        hub_url=hub_url,
    )
    model.summary(line_length=90, expand_nested=False)

    # ── Common callbacks ─────────────────────────────────────────────────────
    tensorboard_dir = ckpt_dir / "logs"

    def _callbacks(phase: str, monitor: str = "val_accuracy"):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=7, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=0.5, patience=4, min_lr=1e-7, verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_dir / phase), histogram_freq=0
            ),
        ]

    history_p1 = history_p2 = None

    # ── Phase 1: Train head only ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"Phase 1 — head only  (lr=1e-3, max {phase1_epochs} epochs)")
    print("─" * 70)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_p1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=_callbacks("phase1"),
        verbose=1,
    )
    best_val_p1 = max(history_p1.history.get("val_accuracy", [0.0]))
    print(f"\nPhase 1 best val_accuracy: {best_val_p1:.4f}")

    # ── Phase 2: Unfreeze top backbone layers ─────────────────────────────────
    if phase2_epochs > 0 and unfreeze_top > 0:
        print("\n" + "─" * 70)
        print(f"Phase 2 — fine-tune top {unfreeze_top} backbone layers  "
              f"(lr=1e-5, max {phase2_epochs} epochs)")
        print("─" * 70)

        unfreeze_top_layers(model, n_layers=unfreeze_top)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history_p2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=phase2_epochs,
            callbacks=_callbacks("phase2"),
            verbose=1,
        )
        best_val_p2 = max(history_p2.history.get("val_accuracy", [0.0]))
        print(f"\nPhase 2 best val_accuracy: {best_val_p2:.4f}")

    # ── Evaluate on test set ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Test set evaluation")
    print("─" * 70)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"  test_loss     : {test_loss:.4f}")
    print(f"  test_accuracy : {test_acc:.4f}")

    # ── Save model ───────────────────────────────────────────────────────────
    model_path = ckpt_dir / "model.keras"
    model.save(str(model_path))
    print(f"\nModel saved → {model_path}")

    # ── Write training log ───────────────────────────────────────────────────
    log = {
        "run_name":      run_name,
        "timestamp":     ts,
        "dataset":       str(dataset_dir),
        "classes":       classes,
        "num_classes":   num_classes,
        "batch_size":    batch_size,
        "phase1_epochs": len(history_p1.epoch) if history_p1 else 0,
        "phase2_epochs": len(history_p2.epoch) if history_p2 else 0,
        "unfreeze_top":  unfreeze_top,
        "test_accuracy": float(test_acc),
        "test_loss":     float(test_loss),
        "model_path":    str(model_path),
        "class_map_path": str(class_map_path),
    }
    log_path = ckpt_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log  → {log_path}")

    # ── Register in model_store/registry.json ───────────────────────────────
    _update_registry(
        registry_path = REPO_ROOT / "model_store" / "registry.json",
        run_name      = run_name,
        classes       = classes,
        test_accuracy = float(test_acc),
        model_path    = str(model_path),
        dataset       = str(dataset_dir),
        timestamp     = ts,
    )

    print("\n" + "=" * 70)
    print(f"Training complete.  Checkpoint: {ckpt_dir}")
    print(f"Next step: python training/export_finetuned.py --checkpoint {ckpt_dir}")
    print("=" * 70)

    return ckpt_dir


def _update_registry(
    registry_path: Path,
    run_name: str,
    classes: list,
    test_accuracy: float,
    model_path: str,
    dataset: str,
    timestamp: str,
) -> None:
    """Append or update an entry in model_store/registry.json."""
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"schema_version": "1", "models": [], "active_model": None}

    # Overwrite entry if same run_name already exists
    registry["models"] = [
        m for m in registry["models"] if m.get("run_name") != run_name
    ]
    registry["models"].append({
        "run_name":        run_name,
        "timestamp":       timestamp,
        "classes":         classes,
        "num_classes":     len(classes),
        "val_accuracy":    test_accuracy,
        "model_path":      model_path,
        "tflite_path":     None,   # filled by export_finetuned.py
        "tflite_int8_path": None,
        "dataset":         dataset,
        "deployed":        False,
    })

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Registry updated  → {registry_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune YAMNet on a custom wildlife dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",       required=True,
                   help="Path to curator dataset (contains audio/ and labels.csv)")
    p.add_argument("--savedmodel",    default=None,
                   help="Path to yamnet_core/ SavedModel (preferred weight source)")
    p.add_argument("--hub-url",       default="https://tfhub.dev/google/yamnet/1",
                   help="TFHub URL fallback if --savedmodel is absent")
    p.add_argument("--phase1-epochs", type=int, default=20)
    p.add_argument("--phase2-epochs", type=int, default=30,
                   help="Set to 0 to skip Phase 2 fine-tuning")
    p.add_argument("--unfreeze-top",  type=int, default=4,
                   help="Number of top backbone layers to unfreeze in Phase 2")
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--output-dir",    default=str(MODELS_DIR),
                   help="Parent directory for checkpoint runs")
    p.add_argument("--run-name",      default=None,
                   help="Override the auto-generated run name")
    return p.parse_args()


if __name__ == "__main__":
    # Make relative imports work when run as a script
    sys.path.insert(0, str(SCRIPT_DIR))
    # Add integration/ so export helpers are importable
    sys.path.insert(0, str(REPO_ROOT / "integration"))

    args = _parse_args()
    train(
        dataset_dir     = args.dataset,
        savedmodel_path = args.savedmodel,
        hub_url         = args.hub_url,
        phase1_epochs   = args.phase1_epochs,
        phase2_epochs   = args.phase2_epochs,
        unfreeze_top    = args.unfreeze_top,
        batch_size      = args.batch_size,
        output_dir      = args.output_dir,
        run_name        = args.run_name,
    )
