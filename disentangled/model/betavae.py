import tensorflow as tf

import disentangled.model.conv as conv
import disentangled.model.objectives as objectives


class Encoder(tf.keras.layers.Layer):
    def __init__(self, latents=32, **kwargs):
        super(Encoder, self).__init__(**kwargs)
         
        self.f_theta = tf.keras.Sequential([ 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(400, activation="relu"),
            tf.keras.layers.Dense(2*latents, activation=None)
            ])

    def call(self, x):
        x = self.f_theta(x)

        return tf.split(x, 2, axis=1)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_dim=(28,28,1), **kwargs):
        super(Decoder, self).__init__(**kwargs)
 
        self.f_phi = tf.keras.Sequential([ 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(400, activation="relu"),
            tf.keras.layers.Dense(2*tf.reduce_prod(output_dim), activation=None),
            ])

    def call(self, x):
        x = self.f_phi(x)
        return tf.split(x, 2, axis=1)

class BetaVAE(tf.keras.Model):
    def __init__(
        self, output_dim=(28,28,1), encoder=Encoder(), decoder=Decoder(), latents=32, **kwargs
        ):
        super(BetaVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latents = latents

        self.reshape = tf.keras.layers.Reshape(output_dim)


    def sample(self, mean, log_var, training):
        if not training:
            return mean

        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)

        return mean + tf.exp(0.5 * log_var) * noise

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def call(self, x, training=False):
        z_mean, z_log_var = self.encode(x)
        z = self.sample(z_mean, z_log_var, training)
        x_mean, x_log_var = self.decoder(z)
        return self.reshape(x_mean) 
