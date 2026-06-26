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

# ─────────────────────────────────────────────────────────────────────────────
# Focal loss
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(gamma: float = 2.0, alpha: "dict | None" = None):
    """
    Alpha-weighted multi-class focal loss.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0, alpha=None  -> plain categorical cross-entropy
    gamma=2              -> standard RetinaNet default; down-weights easy examples
    alpha                -> per-class inverse-frequency weights {class_idx: weight}
                           Up-weights rare class (e.g. elephant) so its errors cost
                           more than easy background samples.

    Using BOTH gamma>0 AND alpha is the recommended setting for severe class
    imbalance: alpha corrects the prior, gamma corrects the easy/hard imbalance.

    Expects one-hot encoded labels (same format as categorical_crossentropy).
    """
    # Build a constant weight vector once, outside the hot path
    alpha_vec = None
    if alpha:
        max_idx   = max(alpha.keys())
        alpha_arr = [alpha.get(i, 1.0) for i in range(max_idx + 1)]
        alpha_vec = tf.constant(alpha_arr, dtype=tf.float32)   # (C,)

    def _loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)                      # (B, C)
        p_t    = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)  # (B,1)
        fl     = tf.pow(1.0 - p_t, gamma) * ce                     # (B, C)
        if alpha_vec is not None:
            fl = fl * tf.reshape(alpha_vec, (1, -1))                # alpha per class
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))           # scalar

    _loss.__name__ = f"focal_loss_g{gamma}_alpha{'yes' if alpha else 'no'}"
    return _loss


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
    """Transfer backbone weights from a local yamnet_core SavedModel.

    Uses tf.saved_model.load() directly — no tensorflow_hub dependency.
    The SavedModel variables are named without a model-name prefix
    (e.g. "layer1/conv/kernel:0"), while the finetuned model variables
    carry a "yamnet_finetuned/" prefix that we strip before matching.
    """
    try:
        saved = tf.saved_model.load(savedmodel_path)

        # Build name → variable map; strip trailing ":0" for lookup keys
        src_by_name: dict = {}
        for v in saved.variables:
            key = v.name[:-2] if v.name.endswith(":0") else v.name
            src_by_name[key] = v

        # Fragments that identify head-only variables (must NOT be overwritten)
        head_keys = {"head_fc", "head_dropout", "custom_predictions",
                     "logits", "predictions"}

        copied = 0
        for var in model.variables:
            # Skip any variable that belongs to the custom head
            if any(k in var.name for k in head_keys):
                continue
            # Strip ":0" suffix to get the lookup key
            name = var.name
            if name.endswith(":0"):
                name = name[:-2]

            # Strategy 1: exact match (tf_keras/Keras-2 — no model-name prefix)
            if name in src_by_name and src_by_name[name].shape == var.shape:
                var.assign(src_by_name[name])
                copied += 1
                continue

            # Strategy 2: strip one-level model-name prefix (Keras-3 style:
            #   "yamnet_finetuned/layer1/conv/kernel" → "layer1/conv/kernel")
            parts = name.split("/", 1)
            short = parts[1] if len(parts) > 1 else parts[0]
            if short in src_by_name and src_by_name[short].shape == var.shape:
                var.assign(src_by_name[short])
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

def _load_from_keras_checkpoint(
    model           : tf.keras.Model,
    checkpoint_path : str,
) -> int:
    """Warm-start from an existing fine-tuned model.keras.

    Transfers all variables with matching names AND shapes.
    custom_predictions is automatically skipped when num_classes differs,
    so this works whether the new dataset has the same or different class count.

    Returns the number of variables copied.
    """
    try:
        ckpt = tf.keras.models.load_model(checkpoint_path, compile=False)
        src  = {v.name: v for v in ckpt.variables}
        copied = skipped_shape = skipped_missing = 0
        for var in model.variables:
            s = src.get(var.name)
            if s is None:
                skipped_missing += 1
            elif s.shape != var.shape:
                skipped_shape += 1
            else:
                var.assign(s)
                copied += 1
        print(f"  ✓ Warm-start: {copied} variables transferred, "
              f"{skipped_shape} shape-mismatch (new head ok), "
              f"{skipped_missing} not in source")
        return copied
    except Exception as exc:
        print(f"  ✗ Warm-start load failed: {exc}")
        return 0




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
    focal_gamma:    float = 0.0,
    weight_floor:   float = 0.0,
    warm_start_path : str | None = None,
) -> Path:
    """
    Full two-phase training pipeline.

    Returns the path to the output checkpoint directory.
    """
    # Import here so the module is usable without a full TF install
    from data_loader import load_dataset, compute_class_weights  # relative import within training/

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

    # ── Compute inverse-frequency alpha weights (always) ────────────────
    # Used as alpha in focal loss AND as Keras class_weight when focal is off.
    # For severe imbalance (background >> elephant) these weights up-penalise
    # every missed elephant relative to an easy background sample.
    print("\nComputing class weights ...")
    class_weight = compute_class_weights(dataset_dir, weight_floor=weight_floor)
    if not class_weight:
        print("  [WARNING] No class weights computed — training without weighting")

    # ── Loss function selection ──────────────────────────────────────────────────────
    # focal_gamma=2.0 is recommended for wildlife detection (rare class vs background).
    # It activates BOTH alpha weighting (corrects class prior) AND focal modulation
    # (down-weights easy background samples so the model focuses on rare events).
    # Alpha is embedded inside the loss; Keras class_weight is cleared to avoid
    # double-counting.
    if focal_gamma > 0.0:
        loss_fn      = focal_loss(gamma=focal_gamma, alpha=class_weight)
        class_weight = {}   # alpha already inside loss; don't double-count
        print(f"\nUsing alpha-weighted focal loss  (gamma={focal_gamma}, alpha=per-class inverse-freq)")
    else:
        loss_fn = "categorical_crossentropy"
        print("\nUsing categorical cross-entropy with Keras class_weight")

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

    # ── Warm-start from an existing checkpoint (optional) ────────────────────
    if warm_start_path and Path(warm_start_path).exists():
        print(f"\nWarm-starting backbone from: {warm_start_path}")
        n = _load_from_keras_checkpoint(model, warm_start_path)
        if n == 0:
            print("  ⚠ No variables transferred — using pretrained YAMNet weights")
    elif warm_start_path:
        print(f"\n⚠ Warm-start path not found: {warm_start_path} — ignoring")

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

    # Per-class recall — val_recall_cls1 = "what fraction of real elephants did we catch?"
    recall_metrics = [
        tf.keras.metrics.Recall(class_id=i, name=f"recall_cls{i}")
        for i in range(num_classes)
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=["accuracy"] + recall_metrics,
    )
    history_p1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=_callbacks("phase1", monitor="val_loss"),
        class_weight=class_weight or None,
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
            loss=loss_fn,
            metrics=["accuracy"] + recall_metrics,
        )
        history_p2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=phase2_epochs,
            callbacks=_callbacks("phase2", monitor="val_loss"),
            class_weight=class_weight or None,
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
        "focal_gamma":    focal_gamma,
        "weight_floor":   weight_floor,
        "class_weight":  {classes[i]: round(w, 4) for i, w in class_weight.items()} if class_weight else {},
        "val_recall_per_class": {
            classes[i]: round(float(
                max((history_p2.history.get(f"val_recall_cls{i}") or [])
                    if history_p2 else []
                    or (history_p1.history.get(f"val_recall_cls{i}") or [0]))
            ), 4)
            for i in range(num_classes)
        } if history_p1 else {},
        "warm_start":    str(warm_start_path) if warm_start_path else None,
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

    # Read nickname from meta.json written by yamnet_finetuner.py at launch
    nickname = ""
    meta_path = Path(model_path).parent / "meta.json"
    if meta_path.exists():
        try:
            import json as _json2
            nickname = _json2.loads(meta_path.read_text()).get("nickname", "")
        except Exception:
            pass

    registry["models"].append({
        "run_name":        run_name,
        "nickname":        nickname,
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
    p.add_argument("--focal-gamma",   type=float, default=0.0,
                   help="Gamma for focal loss (0 = disabled, use class weights instead). "
                        "Recommended: 2.0")
    p.add_argument("--warm-start",    default=None,
                   help="Path to a .keras checkpoint to warm-start backbone "
                        "weights from (backbone + head_fc transferred; "
                        "custom_predictions skipped when num_classes differs)")
    p.add_argument("--weight-floor",  type=float, default=0.0,
                   help="Minimum class weight floor — clamps majority-class weights up "
                        "to this value (e.g. 0.5 prevents background being suppressed "
                        "too aggressively). Only used when --focal-gamma=0.")
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
        focal_gamma     = args.focal_gamma,
        weight_floor    = args.weight_floor,
        warm_start_path = args.warm_start,
    )
