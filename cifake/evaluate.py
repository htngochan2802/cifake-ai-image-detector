"""Evaluation helpers: metrics, confusion matrix and training curves."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: figures are saved to disk, not shown
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .data import CLASS_NAMES


def plot_history(history, output_dir):
    """Save loss / accuracy / precision / recall curves (train vs. validation)."""
    output_dir = Path(output_dir)
    metrics = [m for m in ("loss", "accuracy", "precision", "recall") if m in history]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.ravel(), metrics):
        ax.plot(history[metric], label=metric)
        ax.plot(history["val_" + metric], label="val_" + metric)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def evaluate_model(model, test_ds, output_dir):
    """Evaluate on the (unshuffled) test set, save metrics.json + confusion matrix."""
    output_dir = Path(output_dir)
    results = model.evaluate(test_ds, return_dict=True, verbose=1)

    y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int)
    y_prob = model.predict(test_ds, verbose=0).ravel()  # P(REAL)
    y_pred = (y_prob > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   digits=4, output_dict=True, zero_division=0)
    summary = {
        "keras_evaluate": {k: float(v) for k, v in results.items()},
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues")
    plt.title("Confusion matrix (test set)")
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    return summary


def write_comparison(rows, output_dir):
    """Save a Markdown/JSON table comparing the trained models. Returns the Markdown text."""
    output_dir = Path(output_dir)
    lines = ["| Model | Params (M) | Epochs | Train (min) | Test acc | Precision | Recall | AUC |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['params_M']:.2f} | {r['epochs_run']} | {r['train_min']:.1f} "
            f"| {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['auc']:.4f} |")
    text = "\n".join(lines) + "\n"
    (output_dir / "comparison.md").write_text(text)
    (output_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    return text
