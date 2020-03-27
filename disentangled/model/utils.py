from pathlib import Path

import tensorflow as tf


def save(model, filename, path=Path("./models"), overwrite=False, **kwargs):
    path.mkdir(exist_ok=True)
    path = path / filename
    model.save(str(path), overwrite=overwrite)


def load(filename, path=Path("./models")):
    path = path / filename

    return tf.keras.models.load_model(str(path), compile=False)
