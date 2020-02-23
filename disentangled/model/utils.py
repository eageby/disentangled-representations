from pathlib import Path

import tensorflow as tf


def save(model, filename, dir_="models", **kwargs):
    path = Path(".") / dir_
    path.mkdir(exist_ok=True)
    path = path / filename
    model.save(str(path))


def load(filename, dir_="models"):
    path = Path(".")/ dir_ / (filename + filetype) 

    return tf.keras.models.load_model(str(path))
