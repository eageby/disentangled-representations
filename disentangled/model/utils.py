from pathlib import Path

import tensorflow as tf


def save(model, filename, dir_="models", overwrite=False, **kwargs):
    path = Path(".") / dir_
    path.mkdir(exist_ok=True)
    path = path / filename
    model.save(str(path), overwrite=overwrite)


def load(filename, dir_=Path("./models")):
    path = dir_ / filename

    return tf.keras.models.load_model(str(path), compile=False)
