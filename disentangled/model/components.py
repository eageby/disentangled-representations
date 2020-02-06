import tensorflow as tf

class Mlp(tf.keras.layers.Layer):

    def __init__(self, layer_width, latent_dim, **kwargs):
        super(Mlp, self).__init__(**kwargs)

        self.layer_width = layer_width
        self.latent_dim = latent_dim

    def build(self, input_dim):
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(self.layer_width[0])
        self.layer2 = tf.keras.layers.Dense(self.layer_width[1])
        self.layer3 = tf.keras.layers.Dense(self.latent_dim)

    def call(self, input_):
        x = self.flatten(input_)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)
      
class DeMlp(tf.keras.layers.Layer):
    def __init__(self, layer_width, output_dim, **kwargs):

        super(DeMlp, self).__init__(**kwargs)

        self.layer_width = layer_width
        self.output_dim = output_dim
   
    def build(self, input_dim):
        self.layer1 = tf.keras.layers.Dense(self.layer_width[0], input_shape = input_dim)
        self.layer2 = tf.keras.layers.Dense(self.layer_width[1])
        self.layer3 = tf.keras.layers.Dense(tf.reduce_prod(self.output_dim))
        self.reshape = tf.keras.layers.Reshape(self.output_dim)

    def call(self, input_):
        x = self.layer1(input_)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.reshape(x)

class Latent(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Latent, self).__init__(**kwargs)

    def build(self, input_dim):
        self.dim = input_dim

    def call(self, input_, training=False):
        # noise = tf.random.normal(
        #     shape=[self.dim],
            # mean=0.0,
            # stddev=tf.ones([self.dim]),
            # dtype=tf.float32,
        # )  
        return input_

