import tensorflow as tf

import disentangled.model.conv as conv
import disentangled.model.objectives as objectives

from .vae import Representation


class VAE(tf.keras.Model):
    def __init__(
        self, encoder, representation, decoder, objective, latents=20, **kwargs
    ):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.representation = representation
        self.decoder = decoder
        self.objective = objective
        self.latents = latents

    def build(self, input_dim):
        self.flatten = tf.keras.layers.Flatten()
        self.output_reshape = tf.keras.layers.Reshape(input_dim[1:])

        encoder_output_dim = self.encoder.compute_output_shape(input_dim)

        self.dense_mean = tf.keras.layers.Dense(self.latents, activation="relu")
        self.dense_log_var = tf.keras.layers.Dense(self.latents, activation="relu")

        self.representation_dense_out = tf.keras.layers.Dense(
            self.flatten.compute_output_shape(encoder_output_dim)[1], activation="relu"
        )
        self.reshape = tf.keras.layers.Reshape(encoder_output_dim[1:])

    def call(self, inputs, training=False):
        x = inputs
        x = self.encoder(x)
        x = self.flatten(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.representation((z_mean, z_log_var), training)
        x = self.reshape(self.representation_dense_out(z))
        x_mean, x_log_var = self.decoder(x)
        self.add_loss(self.objective((inputs, x_mean, x_log_var, z_mean, z_log_var)))

        return x_mean, z, inputs


class AEVB(VAE):
    def __init__(self, latents, **kwargs):
        super(AEVB, self).__init__(
            conv.Encoder(),
            Representation(),
            conv.Decoder(),
            objectives.BetaVAE(gaussian=True),
        )
