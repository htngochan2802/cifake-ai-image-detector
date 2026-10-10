"""Fast smoke tests (no dataset download, no pretrained weights)."""
import json

import numpy as np
import pytest
from PIL import Image

from cifake import predict as predict_mod
from cifake import train as train_mod
from cifake.data import CLASS_NAMES
from cifake.models import MODEL_NAMES, build_model


def _make_fake_dataset(root, n=12):
    rng = np.random.default_rng(0)
    paths = []
    for split in ("train", "test"):
        for cls in CLASS_NAMES:
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{i}.png")
                paths.append(d / f"{i}.png")
    return paths


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_forward_shape(name):
    model = build_model(name, weights=None)
    out = model.predict(np.random.rand(4, 32, 32, 3).astype("float32") * 255, verbose=0)
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()


def test_train_compare_and_predict_end_to_end(tmp_path):
    images = _make_fake_dataset(tmp_path / "data")
    out = tmp_path / "out"
    train_mod.main(["--data_dir", str(tmp_path / "data"), "--models", "mobilenetv3small",
                    "--weights", "none", "--epochs", "1", "--batch_size", "8",
                    "--val_split", "0.25", "--output_dir", str(out)])

    model_dir = out / "mobilenetv3small"
    for f in ("metrics.json", "config.json", "history.json", "training_curves.png",
              "confusion_matrix.png", "mobilenetv3small_cifake.keras"):
        assert (model_dir / f).exists(), f
    assert (out / "comparison.md").exists()
    assert json.loads((model_dir / "metrics.json").read_text())["roc_auc"] >= 0

    preds = predict_mod.predict_images(model_dir, images[:3])
    assert len(preds) == 3
    assert all(p["label"] in CLASS_NAMES and 0 <= p["p_real"] <= 1 for p in preds)
