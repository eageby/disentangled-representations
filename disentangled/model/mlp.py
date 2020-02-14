import tensorflow as tf
import disentangled.model.noise as noise


class MLP(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layer = tf.keras.layers.Dense(output_dim, activation='relu')

    def call(self, x):
        return self.layer(x)

class DeMLP(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(DeMLP, self).__init__(**kwargs)
        self.layer = tf.keras.layers.Dense(output_dim, activation='relu')

    def call(self, x):
        return self.layer(x)
