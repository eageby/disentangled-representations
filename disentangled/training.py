
import os

from decouple import config

@gin.configurable
def run_training(
    model, dataset, iterations, save=False, callbacks=[], seed=None
) -> tf.keras.Model:
    tf.random.set_seed(seed)
    model.predict(dataset, steps=1)  # Instantiating model
    model.train(dataset.repeat(), callbacks=callbacks, iterations=iterations)

    if save:
        disentangled.model.utils.save(model, gin.REQUIRED)

    return model
