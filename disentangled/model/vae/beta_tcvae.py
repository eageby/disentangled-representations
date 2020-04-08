import tensorflow as tf
from .vae import VAE
from .betavae import BetaVAE

import disentangled.model.networks as networks
import disentangled.utils
import disentangled.model.objectives as objectives

class Beta_TCVAE(BetaVAE):
    pass

class beta_tcvae_shapes3d(Beta_TCVAE):
    def __init__(self, latents, beta, dataset_size, **kwargs):
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
            objective=objectives.Beta_TCVAE(beta=beta, dataset_size=dataset_size),
            latents=latents
        )
