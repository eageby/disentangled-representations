import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.objectives as objectives
import disentangled.utils
import tensorflow as tf

from .vae import VAE

class BetaVAE(VAE):
    def __init__(self, beta, **kwargs):
        super().__init__(
            prior_dist=dist.Gaussian(mean=0.0, log_var=0.0),
            output_dist=dist.Bernoulli(),
            objective=objectives.BetaVAE(),
            name="BetaVAE",
            **kwargs
        )
        self.beta = beta

class betavae_mnist(BetaVAE):
    def __init__(self, latents, beta, **kwargs):
        super().__init__(
            # Encoder
            f_phi=networks.conv_2,
            f_phi_mean=tf.keras.layers.Dense(latents, activation=None),
            f_phi_log_var=tf.keras.layers.Dense(latents, activation=None),
            # Decoder
            f_theta=networks.conv_2_transpose,
            f_theta_mean=tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
            ),
            f_theta_log_var=tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation=None
            ),
            objective=objectives.BetaVAE(gaussian=False, beta=beta),
            latents=latents,
        )


class betavae_shapes3d(BetaVAE):
    def __init__(self, latents, beta, **kwargs):
        super().__init__(
            # Encoder
            f_phi=networks.conv_4,
            f_phi_mean=tf.keras.layers.Dense(latents, activation=None),
            f_phi_log_var=tf.keras.layers.Dense(latents, activation=None),
            # Decoder
            f_theta=networks.conv_4_transpose,
            f_theta_mean=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
            ),
            f_theta_log_var=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation=None
            ),
            latents=latents,
            beta=beta,
        )
