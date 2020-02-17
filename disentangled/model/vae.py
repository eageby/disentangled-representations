import tensorflow as tf
import disentangled.model.objective as objective

class Representation(tf.keras.layers.Layer):
    def call(self, inputs, training=False):
        mean, log_var = inputs
        if not training:
            return mean

        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, seed=10)
        return mean + tf.exp(0.5 * log_var) * noise
                

class Encoder(tf.keras.layers.Layer):
    def __init__(self, latents, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(400, activation='relu')
        self.dense1 = tf.keras.layers.Dense(latents, activation='relu')
        self.repr = Representation()
        
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense1(x) 
        z_mean, z_log_var = tf.split(x, 2, axis=-1)
        return z_mean, z_log_var
        
class Decoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(400, activation='relu') 
        self.dense1 = tf.keras.layers.Dense(2*tf.reduce_prod(input_dim), activation='sigmoid')
       
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dense1(x)
        x_mean, x_log_var = tf.split(x, 2, axis=1)
        return x_mean, x_log_var

class VAE(tf.keras.Model):
    def __init__(self, encoder, representation, decoder, latents=20,**kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latents = latents
        self.encoder = encoder
        self.representation = representation
        self.decoder = decoder

    def build(self, input_dim):
        self.flatten = tf.keras.layers.Flatten()
        self.encoder = self.encoder(self.latents)
        self.decoder = self.decoder(input_dim[1:])

        self.reshape = tf.keras.layers.Reshape(input_dim[1:])
        self.repr = self.representation()
        self.objective = objective.BetaVAE(gaussian=False)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        z_mean, z_log_var = self.encoder(x)
        z = self.repr((z_mean, z_log_var), training)
        x_mean, x_log_var = self.decoder(x)
        self.add_loss(
                self.objective(x, x_mean, x_log_var, z_mean, z_log_var)
                )

        return self.reshape(x_mean)
