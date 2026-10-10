"""Data loading for the CIFAKE dataset."""
from pathlib import Path

import tensorflow as tf

KAGGLE_ID = "birdy654/cifake-real-and-ai-generated-synthetic-images"
IMG_SIZE = (32, 32)

# image_dataset_from_directory sorts folders alphabetically: FAKE -> 0, REAL -> 1.
# The model's sigmoid output is therefore P(image is REAL).
CLASS_NAMES = ["FAKE", "REAL"]


def resolve_data_dir(data_dir=None):
    """Return the CIFAKE root folder (the one containing train/ and test/).

    If ``data_dir`` is None the dataset is downloaded with kagglehub.
    On Kaggle notebooks use /kaggle/input/cifake-real-and-ai-generated-synthetic-images
    """
    if data_dir is None:
        import kagglehub

        data_dir = kagglehub.dataset_download(KAGGLE_ID)
    root = Path(data_dir)
    for split in ("train", "test"):
        if not (root / split).is_dir():
            raise FileNotFoundError(f"Expected a '{split}/' folder inside {root}")
    return root


def load_datasets(data_dir=None, batch_size=500, val_split=0.1, seed=512, preprocess_fn=None):
    """Build the train / validation / test ``tf.data`` pipelines.

    The validation set is carved out of the *training* folder, so the official
    test set is only used for the final evaluation (no leakage through early stopping).
    The test set is not shuffled, so predictions line up with labels.
    """
    root = resolve_data_dir(data_dir)
    common = dict(image_size=IMG_SIZE, batch_size=batch_size, seed=seed)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(root / "train"), validation_split=val_split, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(root / "train"), validation_split=val_split, subset="validation", **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(root / "test"), shuffle=False, **common)

    for ds in (train_ds, val_ds, test_ds):
        if ds.class_names != CLASS_NAMES:
            raise ValueError(f"Unexpected classes {ds.class_names}, expected {CLASS_NAMES}")

    def prepare(ds):
        if preprocess_fn is not None:
            ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    return prepare(train_ds), prepare(val_ds), prepare(test_ds)
