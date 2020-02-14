import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(self, encoder, representation, decoder, objective, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # Layers
        self.flatten = tf.keras.layers.Flatten()
        self.encoder = encoder
        self.representation = representation
        self.decoder = decoder
        self.objective = objective

        self.dense2 = tf.keras.layers.Dense(2*28*28,activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape((28,28,1))

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.encoder(x)  
        z = self.representation(x)
        x = self.decoder(z)
        x_mean, self.x_var = tf.split(self.dense2(x), 2, axis=-1)
        return self.reshape(x_mean)

    def loss(self, predicted, target):
        return self.objective(self.flatten(target), self.flatten(predicted), self.x_var, self.representation.mean, self.representation.var)


class Representation(tf.keras.layers.Layer):
    def __init__(self, latents, **kwargs):
        super(Representation, self).__init__(**kwargs)
        self.latents = latents

    def build(self, input_dim):
        self.flatten = tf.keras.layers.Flatten()
        self.encode = tf.keras.layers.Dense(2*self.latents, activation='relu', name='123')

        self.decode = tf.keras.layers.Dense(self.flatten.compute_output_shape(input_dim)[-1], activation='relu')

    def sample(self, mean, log_var, training=False):
        # if not training:
        #     return mean

        # TODO Make seed globally configured
        noise = tf.random.normal(tf.shape(log_var), mean=0.0, stddev=1.0, seed=10)
        
        return mean + tf.sqrt(tf.exp(log_var)) * noise

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.encode(x)
        self.mean, self.var = tf.split(x,2, axis=-1)
        x = self.sample(self.mean, self.var, training=training)
        return self.decode(x)


class Objective:
    """Abstract Base Class"""
    def objective(self, *args):
        pass

    def __call__(self, *args):
        return self.objective(*args)

class BetaVAE(Objective):
    def __init__(self):
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def kld(self, z_mean, z_var):
        return tf.reduce_mean(-0.5 * tf.reduce_sum(
                1 + z_var - tf.square(z_mean) - tf.exp(z_var)
                ,axis=-1)
                ,axis=0)

    def objective(self, target, x_mean, x_log_var, z_mean, z_var):
        return self.bce(x_mean, target) + self.kld(z_mean, z_var)

