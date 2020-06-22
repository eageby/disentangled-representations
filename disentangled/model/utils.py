from pathlib import Path

import gin
import tensorflow as tf
from decouple import config

import disentangled.utils

@gin.configurable(module="disentangled.model.utils")
def save(model, filename, overwrite=False):
    path = disentangled.utils.get_data_path() / "models"
    path.mkdir(exist_ok=True)
    path = path / filename
    breakpoint()
    model.save(str(path), overwrite=overwrite)


@gin.configurable(module="disentangled.model.utils")
def load(filename):
    path = disentangled.utils.get_data_path() / "models"
    path = path / filename

    return tf.keras.models.load_model(str(path), compile=False)
