import tensorflow as tf

import disentangled.model.conv as conv
import disentangled.model.objectives as objectives

from .vae import *


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
        self.output_reshape = tf.keras.layers.Reshape(input_dim[1:])

    def call(self, inputs, training=False):
        x = inputs
        x = self.encoder(x)
        z, z_mean, z_log_var = self.representation(x)
        x_mean, x_log_var = self.decoder(z)
        self.add_loss(self.objective((inputs, x_mean, x_log_var, z_mean, z_log_var)))

        return self.output_reshape(x_mean), z, inputs


class AEVB(VAE):
    def __init__(self, latents, **kwargs):
        super(AEVB, self).__init__(
            conv.Encoder(),
            Representation(latents),
            conv.Decoder(),
            objectives.BetaVAE(gaussian=True),
        )

class Mobile(VAE):
    def __init__(self, latents, **kwargs):
        super(Mobile, self).__init__(
            conv.MobilenetV2(),
            Representation(),
            conv.Decoder(),
            objectives.BetaVAE(gaussian=True),
        )

class MLP(VAE):
    def __init__(self, latents, **kwargs):
        super(MLP, self).__init__(
            Encoder(),
            Representation(latents),
            Decoder(),
            objectives.BetaVAE(gaussian=False),
        )

