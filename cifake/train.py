"""Train and evaluate transfer-learning models on CIFAKE.

Usage:
    cifake-train --models efficientnetv2b0 --data_dir /path/to/cifake
    cifake-train --models all     # ResNet50, VGG16, EfficientNetV2B0, MobileNetV3Small
    cifake-train --models mobilenetv3small --epochs 2 --weights none   # quick smoke run
(equivalent: python -m cifake.train ...). Without --data_dir the dataset is fetched via kagglehub.
"""
import argparse
import json
import time
from pathlib import Path

import tensorflow as tf

from .data import CLASS_NAMES, IMG_SIZE, load_datasets, resolve_data_dir
from .evaluate import evaluate_model, plot_history, write_comparison
from .models import MODEL_NAMES, build_model, compile_model, default_optimizer, get_preprocess


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"], choices=[*MODEL_NAMES, "all"])
    p.add_argument("--data_dir", default=None, help="CIFAKE root containing train/ and test/")
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--optimizer", choices=["adam", "adamax"], default=None,
                   help="override the per-model default (adamax; adam for mobilenetv3small)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_split", type=float, default=0.1,
                   help="fraction of train used for validation")
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience")
    p.add_argument("--seed", type=int, default=512)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--plot_model", action="store_true",
                   help="save architecture diagrams (needs pydot + graphviz)")
    return p.parse_args(argv)


def run_training(name, args, data_dir):
    """Train one model, evaluate on the test set, save artifacts. Returns a summary row."""
    out = Path(args.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    print(f"\n########## {name} ##########")

    train_ds, val_ds, test_ds = load_datasets(
        data_dir, args.batch_size, args.val_split, args.seed, preprocess_fn=get_preprocess(name))

    model = build_model(name, weights=None if args.weights == "none" else args.weights,
                        seed=args.seed)
    optimizer = args.optimizer or default_optimizer(name)
    compile_model(model, optimizer, args.lr)
    model.summary()
    if args.plot_model:
        try:
            tf.keras.utils.plot_model(
                model, show_shapes=True, to_file=str(out / "architecture.png"))
        except Exception as e:  # pydot/graphviz are optional
            print("Skipping plot_model:", e)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True)
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        verbose=1, callbacks=[early_stopping])
    train_min = (time.time() - t0) / 60

    hist = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, out)

    summary = evaluate_model(model, test_ds, out)
    model.save(out / f"{name}_cifake.keras")
    (out / "config.json").write_text(json.dumps({
        "model": name, "weights": args.weights, "optimizer": optimizer, "lr": args.lr,
        "batch_size": args.batch_size, "val_split": args.val_split, "seed": args.seed,
        "img_size": list(IMG_SIZE), "class_names": CLASS_NAMES,
        "note": "sigmoid output = P(REAL); inputs must go through the model's preprocess_input",
    }, indent=2))

    r = summary["keras_evaluate"]
    return {"model": name, "params_M": model.count_params() / 1e6,
            "epochs_run": len(hist["loss"]), "train_min": train_min,
            "accuracy": r["accuracy"], "precision": r["precision"], "recall": r["recall"],
            "auc": summary["roc_auc"]}


def main(argv=None):
    args = parse_args(argv)
    names = MODEL_NAMES if "all" in args.models else list(dict.fromkeys(args.models))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print("GPUs available:", len(tf.config.list_physical_devices("GPU")))
    data_dir = resolve_data_dir(args.data_dir)

    rows = []
    for name in names:
        rows.append(run_training(name, args, data_dir))
        tf.keras.backend.clear_session()

    table = write_comparison(rows, out_root)
    print("\n" + table)
    print(f"Artifacts saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
