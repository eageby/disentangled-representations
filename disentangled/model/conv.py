import tensorflow as tf
import disentangled.model.activations as activations

class MobilenetV2(tf.keras.layers.Layer):

    def build(self, input_dim):
        self.trainable = False
        self.model = tf.keras.applications.MobileNetV2(input_shape=input_dim, include_top=False, pooling=None, weights='imagenet')
    
    def call(self, x):
        return self.model(x)

class Encoder(tf.keras.Sequential):
    def __init__(self, **kwargs):
        layers = [
            tf.keras.layers.Conv2D(
                32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
            # tf.keras.layers.Conv2D(
            #     64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            # ),
            # tf.keras.layers.Conv2D(
            #     64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            # ),
        ]

        super(Encoder, self).__init__(layers, **kwargs)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.layers = [
            tf.keras.layers.Conv2DTranspose(
                32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
            # tf.keras.layers.Conv2DTranspose(
            #     64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            # ),
            # tf.keras.layers.Conv2DTranspose(
            #     64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            # ),
        ]
        self.mean_layer = tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation=activations.relu1
            )

        self.log_var_layer = tf.keras.layers.Conv2DTranspose(
                1, kernel_size=(3, 3), strides=(1, 1), activation="relu"
            )
        
    
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)

        return mean, log_var
