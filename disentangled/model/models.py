import sys
import tensorflow as tf

from . import vae, networks, objectives

__all__ = ["betavae_mnist", "betavae_shapes3d", "get"]


def get(name):
    return getattr(sys.modules[__name__], name)


betavae_mnist = vae.VAE(
    # Encoder
    f_phi=networks.conv_2,
    # Decoder
    f_theta=networks.conv_2_transpose,
    f_theta_mean=tf.keras.layers.Conv2DTranspose(
        1, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
    ),
    f_theta_log_var=tf.keras.layers.Conv2DTranspose(
        1, kernel_size=(3, 3), strides=(1, 1), activation=None
    ),
    objective=objectives.BetaVAE(gaussian=False, beta=4),
    latents=32,
    hyperparameters={'beta': 4}
)

betavae_shapes3d = vae.VAE(
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
    objective=objectives.BetaVAE(gaussian=False),
    latents=32,
    hyperparameters = {'beta': 4}
)
