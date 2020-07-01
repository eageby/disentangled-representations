from pathlib import Path

import gin
import tensorflow as tf
from decouple import config

import disentangled.utils


@gin.configurable(module="disentangled.model.utils")
def save(model, filename, suffix=None, overwrite=False):
    path = disentangled.utils.get_data_path() / "models"
    path.mkdir(exist_ok=True)
    if suffix is not None:
        filename += '_' + str(suffix)

    path = path / filename
    model.save(str(path), overwrite=overwrite)


@gin.configurable(module="disentangled.model.utils")
def load(filename):
    path = disentangled.utils.get_data_path() / "models"
    path = path / filename

    return tf.keras.models.load_model(str(path), compile=False)
