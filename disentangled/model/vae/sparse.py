import disentangled.model.networks as networks
import disentangled.model.objectives as objectives
import disentangled.model.distributions as dist
import disentangled.utils
import tensorflow as tf

from .vae import VAE

class SparseVAE(VAE):
    def __init__(self, latents, beta, gamma, **kwargs):
        super().__init__(
            # Encoder
            f_phi_mean=tf.keras.layers.Dense(
                latents, activation=None, kernel_regularizer=tf.keras.regularizers.l1(1)
            ),
            f_phi_log_var=tf.keras.layers.Dense(
                latents, activation=None, kernel_regularizer=None
            ),
            prior_dist=dist.Laplacian(mean=0., log_var=0.),
            output_dist=dist.Bernoulli(),
            objective=objectives.SparseVAE(),
            latents=latents,
            name="SparseVAE",
            **kwargs
        )
        self.beta = beta
        self.gamma = gamma

    @tf.function
    def sample(self, mean, log_var, training=False):
        gammoid = tf.random.gamma([1], alpha=1, beta=1)

        noise = tf.random.normal(shape=tf.shape(mean), mean=0., stddev=1.)

        return gammoid * mean + tf.math.sqrt(gammoid) * noise * tf.math.exp(0.5*log_var)


class sparsevae_shapes3d(SparseVAE):
    def __init__(self, latents, beta, gamma, **kwargs):
        super().__init__(
            # Encoder
            f_phi=networks.conv_4,
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
            gamma=gamma,
        )
