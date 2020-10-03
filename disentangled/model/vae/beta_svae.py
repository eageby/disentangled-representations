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
    def call(self, target, training=False):
        z_mean, z_log_b2 = self.encode(target)
        z = self.sample(z_mean, z_log_b2, training)
        x_mean, x_log_b2 = self.decode(z)

        return x_mean, z, target


    @tf.function
    def sample(self, mean, log_b2, training=False):
        """sample
        Sampling y from Y ~ Laplacian(μ,b) 
        Z ~ N(0,1)
        V ~ Exponential(1) = Gamma(1,1)
        y = μ + bz(2v)^1/2
        """
        if not training:
            return mean

        # Exponential is special case of Gamma 
        # Exponential(λ) = Gamma(1,λ)
        exponential = tf.random.gamma(tf.shape(mean), alpha=1, beta=1)
        gaussian = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)
        
        return mean + tf.exp(0.5*log_b2)*tf.sqrt(2*exponential)*gaussian
