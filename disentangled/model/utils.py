from pathlib import Path

import gin
import tensorflow as tf
from decouple import config

import disentangled.utils


@gin.configurable(module="disentangled.model.utils")
def save(model, filename, prefix=None, suffix=None, hyperparameter_index=None, random_seed=None, overwrite=False):
    path = disentangled.utils.get_data_path() / "models"

    for i in [prefix, filename, suffix]:
        if i is not None:
            path /= i

    if hyperparameter_index is not None:
        path /= 'HP{}'.format(hyperparameter_index)

    if random_seed is not None:
        path /= 'RS{}'.format(random_seed)

    path.mkdir(exist_ok=True, parents=True)
    model.save(str(path), overwrite=overwrite)
    print('Saved model to {}'.format(str(path.resolve())))


@gin.configurable(module="disentangled.model.utils")
def load(filename):
    path = disentangled.utils.get_data_path() / "models"
    path = path / filename

    return tf.keras.models.load_model(str(path), compile=False)
