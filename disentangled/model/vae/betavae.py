import tensorflow as tf
from .vae import VAE

import disentangled.model.networks as networks
import disentangled.model.objectives as objectives

class BetaVAE(VAE):
    def train(self, data, learning_rate, iterations=100, **kwargs):
        self.compile(tf.keras.optimizers.Adam(learning_rate))
        self.fit(data, steps_per_epoch=iterations)

class betavae_mnist(BetaVAE):
    def __init__(self, latents, beta):
        super().__init__(
            # Encoder
            f_phi=networks.conv_2,
            f_phi_mean= tf.keras.layers.Dense( latents, activation=None),
            f_phi_log_var = tf.keras.layers.Dense( latents, activation=None),
            # Decoder
            f_theta=networks.conv_2_transpose,
            f_theta_mean=tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
            ),
            f_theta_log_var=tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation=None
            ),
            objective=objectives.BetaVAE(gaussian=False, beta=beta),
            latents=latents
        )

class betavae_shapes3d(BetaVAE):
    def __init__(self, latents, beta):
        super().__init__(
            # Encoder
            f_phi=networks.conv_4,
            f_phi_mean= tf.keras.layers.Dense( latents, activation=None),
            f_phi_log_var = tf.keras.layers.Dense( latents, activation=None),
            # Decoder
            f_theta=networks.conv_4_transpose,
            f_theta_mean=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
            ),
            f_theta_log_var=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation=None
            ),
            objective=objectives.BetaVAE(gaussian=False, beta=beta),
            latents=latents
        )
