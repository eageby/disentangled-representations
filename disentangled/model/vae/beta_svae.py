import gin.tf
import tensorflow as tf

import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.objectives as objectives
import disentangled.utils

from .vae import VAE

@gin.configurable(module="model")
class BetaSVAE(VAE):
    def __init__(self, **kwargs):
        super().__init__(name="BetaSVAE", **kwargs)

    @tf.function
    def sample(self, mean, log_var, training=False):
        if not training:
            return mean

        gammoid = tf.random.gamma([1], alpha=1, beta=1)

        noise = tf.random.normal(shape=tf.shape(mean), mean=0.0, stddev=1.0)

        return gammoid * mean + tf.math.sqrt(gammoid) * noise * tf.math.exp(
            0.5 * log_var
        )
