import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.objectives as objectives
import disentangled.utils
import tensorflow as tf

from .betavae import BetaVAE
from .vae import VAE


class Beta_TCVAE(VAE):
    def __init__(self, beta, dataset_size, **kwargs):
        super().__init__(
            prior_dist=dist.Gaussian(mean=0.0, log_var=0.0),
            output_dist=dist.Bernoulli(),
            objective=objectives.Beta_TCVAE(),
            name="Beta_TCVAE",
            **kwargs
        )
        self.beta = beta
        self.dataset_size = dataset_size

class beta_tcvae_shapes3d(Beta_TCVAE):
    def __init__(self, latents, beta, dataset_size, **kwargs):
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
            dataset_size=dataset_size,
        )
