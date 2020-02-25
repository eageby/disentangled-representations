import tensorflow as tf

import disentangled.model.activations as activations

class Representation(tf.keras.layers.Layer):
    def __init__(self, latent_dim, **kwargs):
        super(Representation, self).__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_dim):
        self.flatten = tf.keras.layers.Flatten()
        self.reshape = tf.keras.layers.Reshape(input_dim[1:])
        self.dense_mean = tf.keras.layers.Dense(2*self.latent_dim, activation=None)
        # self.dense_log_var = tf.keras.layers.Dense(self.latent_dim, activation='tanh')
        self.dense_out = tf.keras.layers.Dense(tf.reduce_prod(input_dim[1:]), activation="relu")

    def call(self, input_, training=False):
        x = self.flatten(input_)
        mean, log_var = tf.split(self.dense_mean(x), 2, axis=1)
        # log_var = tf.math.log(self.dense_log_var(x) + 1)
        z = self.sample(mean, log_var, training)
        x = self.dense_out(z)
        return self.reshape(x), mean, log_var
    
    def sample(self, mean, log_var, training):
        if not training:
            return mean

        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, seed=10)

        return mean + tf.exp(0.5 * 1) * noise


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(400, activation="relu")

    def call(self, inputs):
        return self.dense(self.flatten(inputs))

class Decoder(tf.keras.layers.Layer):
    def build(self, input_dim):
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(400, activation="relu")
        self.dense_mean = tf.keras.layers.Dense(28*28, activation='sigmoid')
        self.dense_log_var = tf.keras.layers.Dense(28*28, activation=None)
        
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense(x)
        return self.dense_mean(x), self.dense_log_var(x)
