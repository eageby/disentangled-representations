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
    model, dataset, iterations, save=False, callbacks=[]
) -> tf.keras.Model:
    model.predict(dataset, steps=1)  # Instantiating model
    model.train(dataset.repeat(), callbacks=callbacks, iterations=iterations)

    if save:
        disentangled.model.utils.save(model, gin.REQUIRED)

    return model


if __name__ == "__main__":
    with disentangled.utils.config_path():
        gin.parse_config_file("BetaVAE/Shapes3d.gin")

    gin.bind_parameter("VAE.train.iterations", 10)
    # print(gin.operative_config_str())
    run_training(model=gin.REQUIRED, dataset=gin.REQUIRED)
