import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layer = tf.keras.layers.Dense(output_dim, activation='relu')

    def call(self, x):
        return self.layer(x)

class DeMLP(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(DeMLP, self).__init__(**kwargs)

    def build(self, input_dim):
        self.input_dim = list(input_dim)
        self.input_dim[-1] *= 2

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.input_dim[-1])
        self.reshape = tf.keras.layers.Reshape(self.input_dim[1:])

    def call(self, x):
        x = self.flatten(x) 
        x = self.dense(x)
        x = self.reshape(x)
        return tf.split(x, 2, axis=-1)
