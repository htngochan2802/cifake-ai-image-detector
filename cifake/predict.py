"""Classify images with a trained model.

Usage:
    cifake-predict --model_dir outputs/efficientnetv2b0 path/to/a.png path/to/b.jpg

Images are resized to 32x32 (the CIFAKE resolution) and passed through the same
preprocessing used in training. Output: P(REAL) and the predicted label.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import CLASS_NAMES, IMG_SIZE
from .models import get_preprocess


def load_trained(model_dir):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    model = tf.keras.models.load_model(model_dir / f"{cfg['model']}_cifake.keras")
    return model, cfg


def predict_images(model_dir, image_paths, threshold=0.5):
    model, cfg = load_trained(model_dir)
    preprocess = get_preprocess(cfg["model"])
    batch = np.stack([
        tf.keras.utils.img_to_array(tf.keras.utils.load_img(p, target_size=IMG_SIZE))
        for p in image_paths])
    probs = model.predict(preprocess(batch), verbose=0).ravel()  # P(REAL)
    return [{"path": str(p), "p_real": float(pr),
             "label": CLASS_NAMES[int(pr > threshold)]}  # FAKE=0, REAL=1
            for p, pr in zip(image_paths, probs)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model_dir", required=True,
                    help="folder with config.json and the .keras file")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    for r in predict_images(args.model_dir, args.images, args.threshold):
        print(f"{r['label']:>4}  P(REAL)={r['p_real']:.4f}  {r['path']}")


if __name__ == "__main__":
    main()
