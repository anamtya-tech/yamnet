"""
Confusion matrix evaluation for a trained YAMNet checkpoint.

Usage:
  python confusion_matrix_eval.py \
      --run chatak_yamnet_20260411_101107 \
      [--model_store /home/azureuser/yamnet/model_store/checkpoints] \
      [--out confusion_matrix.png]
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# ── suppress TF noise ────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# ── local imports ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from data_loader import build_label_map, load_patches_for_split


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run",         required=True,
                   help="Run name, e.g. chatak_yamnet_20260411_101107")
    p.add_argument("--model_store", default="/home/azureuser/yamnet/model_store/checkpoints")
    p.add_argument("--out",         default=None,
                   help="Output PNG path (default: <model_store>/<run>/confusion_matrix.png)")
    p.add_argument("--fold",        default="test", choices=["train","val","test"],
                   help="Which fold to evaluate (default: test)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir  = Path(args.model_store) / args.run
    log_path = run_dir / "training_log.json"
    if not log_path.exists():
        sys.exit(f"training_log.json not found in {run_dir}")

    log = json.loads(log_path.read_text())
    dataset_dir = Path(log["dataset"])
    out_path    = Path(args.out) if args.out else run_dir / f"confusion_matrix_{args.fold}.png"

    print(f"Run       : {args.run}")
    print(f"Dataset   : {dataset_dir}")
    print(f"Fold      : {args.fold}")
    print(f"Model     : {run_dir / 'model.keras'}")

    # ── 1. Load class map ────────────────────────────────────────────────────
    labels_csv   = dataset_dir / "labels.csv"
    classes, class_to_idx = build_label_map(labels_csv)
    num_classes  = len(classes)
    print(f"Classes   : {num_classes}  {classes}")

    # ── 2. Load test patches ─────────────────────────────────────────────────
    print(f"\nLoading {args.fold} patches …")
    X_te, y_te = load_patches_for_split(dataset_dir, args.fold, class_to_idx)
    print(f"  Loaded {len(X_te)} patches")

    if len(X_te) == 0:
        sys.exit("No test patches found — check fold labels in labels.csv")

    # ── 3. Load model ────────────────────────────────────────────────────────
    model_path = run_dir / "model.keras"
    print("\nLoading model …")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    # ── 4. Predict ───────────────────────────────────────────────────────────
    print("Running inference …")
    logits = model.predict(X_te, batch_size=64, verbose=0)
    y_pred = np.argmax(logits, axis=1)

    # ── 5. Compute confusion matrix ──────────────────────────────────────────
    from sklearn.metrics import (
        confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
    )

    acc     = accuracy_score(y_te, y_pred)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    cm      = confusion_matrix(y_te, y_pred, labels=list(range(num_classes)))
    report  = classification_report(y_te, y_pred,
                                    target_names=classes,
                                    digits=3, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced accuracy : {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"{'='*60}")
    print("\nPer-class report:")
    print(report)

    # Per-class accuracy table
    print("Per-class accuracy (diagonal / row total):")
    print(f"  {'Class':<20}  {'Correct':>7}  {'Total':>7}  {'Acc':>7}  {'Missed as (top confusions)'}")
    for i, cls in enumerate(classes):
        row_total = cm[i].sum()
        correct   = cm[i, i]
        cls_acc   = correct / row_total if row_total else 0.0
        # top 2 confusions (exclude correct class)
        conf_row  = cm[i].copy(); conf_row[i] = 0
        top_idx   = np.argsort(conf_row)[::-1][:2]
        conf_str  = ", ".join(
            f"{classes[j]} ({conf_row[j]})" for j in top_idx if conf_row[j] > 0
        )
        print(f"  {cls:<20}  {correct:>7}  {row_total:>7}  {cls_acc:>7.1%}  {conf_str}")

    # ── 6. Plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.ticker import MaxNLocator

        # Normalise by row (recall per class)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)

        fig, axes = plt.subplots(1, 2, figsize=(22, 9))
        fig.suptitle(
            f"{args.run}  |  {args.fold} set  |  acc={acc:.3f}  bal_acc={bal_acc:.3f}",
            fontsize=13, fontweight="bold"
        )

        for ax, data, title, fmt, cmap in [
            (axes[0], cm,      "Counts",           "d",    "Blues"),
            (axes[1], cm_norm, "Row-normalised",   ".2f",  "RdYlGn"),
        ]:
            im = ax.imshow(data, interpolation="nearest", cmap=cmap,
                           vmin=0, vmax=(1.0 if fmt==".2f" else None))
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_ylabel("True label", fontsize=10)
            ax.set_xlabel("Predicted label", fontsize=10)

            thresh = data.max() / 2.0 if fmt == "d" else 0.5
            for r in range(num_classes):
                for c in range(num_classes):
                    val = data[r, c]
                    txt = format(int(val), fmt) if fmt == "d" else format(val, fmt)
                    color = "white" if val > thresh else "black"
                    ax.text(c, r, txt, ha="center", va="center",
                            fontsize=6, color=color)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"\nConfusion matrix saved → {out_path}")

    except ImportError as e:
        print(f"\n[WARNING] matplotlib/sklearn not available for plotting: {e}")
        print("Raw confusion matrix (rows=true, cols=predicted):")
        header = "             " + "".join(f"{c[:6]:>8}" for c in classes)
        print(header)
        for i, cls in enumerate(classes):
            row = "".join(f"{cm[i,j]:>8d}" for j in range(num_classes))
            print(f"  {cls:<12} {row}")

    # ── 7. Save text report ───────────────────────────────────────────────────
    txt_path = out_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write(f"Run: {args.run}\n")
        f.write(f"Fold: {args.fold}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced accuracy: {bal_acc:.4f}\n\n")
        f.write(report)
        f.write("\n\nRaw confusion matrix (rows=true, cols=predicted):\n")
        f.write("             " + "".join(f"{c[:7]:>8}" for c in classes) + "\n")
        for i, cls in enumerate(classes):
            row = "".join(f"{cm[i,j]:>8d}" for j in range(num_classes))
            f.write(f"  {cls:<12} {row}\n")
    print(f"Text report saved  → {txt_path}")


if __name__ == "__main__":
    main()
