"""Model zoo: ImageNet backbones + a shared classification head.

All models share the same head (as in the original notebook) so the comparison is fair:
backbone (global max pooling) -> BatchNorm -> Dense(256, L1/L2 regularised) -> Dropout(0.4)
-> Dense(64) -> Dense(1, sigmoid). The sigmoid output is P(image is REAL).

To add a backbone, add one ``ModelSpec`` to ``SPECS``.
"""
from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout

from .data import IMG_SIZE

apps = tf.keras.applications


@dataclass(frozen=True)
class ModelSpec:
    backbone: Callable      # keras.applications constructor
    preprocess: Callable    # matching preprocess_input (identity for EfficientNetV2/MobileNetV3)
    optimizer: str = "adamax"  # default optimizer used in the original notebook


SPECS = {
    "resnet50": ModelSpec(apps.ResNet50, apps.resnet50.preprocess_input),
    "vgg16": ModelSpec(apps.VGG16, apps.vgg16.preprocess_input),
    "efficientnetv2b0": ModelSpec(apps.EfficientNetV2B0, apps.efficientnet_v2.preprocess_input),
    "mobilenetv3small": ModelSpec(apps.MobileNetV3Small, apps.mobilenet_v3.preprocess_input,
                                  optimizer="adam"),
}
MODEL_NAMES = list(SPECS)

OPTIMIZERS = {"adam": tf.keras.optimizers.Adam, "adamax": tf.keras.optimizers.Adamax}


def _spec(name):
    if name not in SPECS:
        raise ValueError(f"Unknown model '{name}'. Available: {MODEL_NAMES}")
    return SPECS[name]


def get_preprocess(name):
    return _spec(name).preprocess


def default_optimizer(name):
    return _spec(name).optimizer


def build_model(name, input_shape=(*IMG_SIZE, 3), weights="imagenet", fine_tune=True, seed=512):
    """Build backbone + head.

    Args:
        weights: "imagenet" for transfer learning, or None to train from scratch.
        fine_tune: if True the backbone weights are updated during training.
    """
    base = _spec(name).backbone(
        include_top=False, weights=weights, input_shape=input_shape, pooling="max")
    base.trainable = fine_tune

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)  # keep the backbone's BatchNorm layers in inference mode
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(256,
              kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l1(0.01),
              bias_regularizer=regularizers.l1(0.01),
              activation="relu")(x)
    x = Dropout(rate=0.4, seed=seed)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name=f"{name}_transfer")


def compile_model(model, optimizer="adamax", learning_rate=1e-3):
    model.compile(
        optimizer=OPTIMIZERS[optimizer](learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")])
    return model
