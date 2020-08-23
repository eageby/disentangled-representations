import disentangled.dataset
import disentangled.model
import disentangled.model.utils
import disentangled.utils
import disentangled.visualize
import tensorflow as tf
import gin.tf

import os

from decouple import config

@gin.configurable
def run_training(
    model, dataset, iterations, save=False, callbacks=[], seed=None, path=None
) -> tf.keras.Model:
    tf.random.set_seed(seed)
    model.predict(dataset, steps=1)  # Instantiating model
    model.train(dataset.repeat(), callbacks=callbacks, iterations=iterations)

    if save:
        disentangled.model.utils.save(model, gin.REQUIRED, path=path)

    return model
