# 
import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, objective, latents=20, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.encoder = tf.keras.layers.Dense(400, activation='relu')   
        self.representation_dense = tf.keras.layers.Dense(2*latents, activation=None)
        self.decoder = tf.keras.layers.Dense(400, activation='relu')   
        self.objective = objective

    def build(self, input_dim):
        self.output_layer = tf.keras.layers.Dense(2*tf.reduce_prod(input_dim[1:]), activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape(input_dim[1:])
        
    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.encoder(x)

        x = self.representation_dense(x)
        self.z_mean, self.z_log_var = tf.split(x, 2, axis=-1)
        z = self.sample(self.z_mean, self.z_log_var, training) 

        x = self.decoder(z) 
        x = self.output_layer(x)
        x_reconstructed, self.x_log_var = tf.split(x, 2, axis=-1)
        return self.reshape(x_reconstructed)

    def loss(self, predicted, target): return self.objective(self.flatten(predicted), self.flatten(target), self.flatten(self.x_log_var), self.z_mean, self.z_log_var)

    def sample(self, mean, log_var, training=False):
        if not training:
            return mean

        # TODO Make seed globally configured
        noise = tf.random.normal(tf.shape(log_var), mean=0.0, stddev=1.0, seed=10)
        
        return mean + tf.exp(0.5 * log_var) * noise

class BetaVAE():
    def __init__(self, gaussian):
        self.gaussian = gaussian

        if not gaussian:
            self.bce = tf.keras.losses.BinaryCrossentropy()

    def __call__(self, *args):
        return self.objective(*args)

    def kld(self, z_mean, z_log_var):
        inner = tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                ,axis=-1)
        return tf.reduce_mean(-0.5 * inner, axis=0)

    def log_likelihood(self, target, x_mean, x_log_var):
        return tf.reduce_mean(-0.5 * 
                   tf.reduce_sum(  
                       x_log_var + tf.square(target - x_mean) / tf.exp(x_log_var)
                   , axis=-1),
               axis=0)
        
    def objective(self, target, x_mean, x_log_var, z_mean, z_log_var):
        if self.gaussian:
            return self.log_likelihood(target, x_mean, x_log_var) + self.kld(z_mean, z_log_var)
        else:
            return self.bce(target, x_mean) + self.kld(z_mean, z_log_var)
