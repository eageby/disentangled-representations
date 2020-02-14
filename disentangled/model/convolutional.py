import tensorflow as tf

class Conv32(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Conv32, self).__init__(**kwargs)

        self.output_dim = 32
        self.layers = [
                tf.keras.layers.Conv2D(
                filters=32, 
                kernel_size=(4,4), 
                strides=(2,2),
                padding='same',
                activation='relu',
                ),
        tf.keras.layers.Conv2D(
                filters=32, 
                kernel_size=(4,4),
                strides=(2,2),
                padding='same',
                activation='relu'
                ),
        tf.keras.layers.Conv2D(
                filters=64, 
                kernel_size=(4,4),
                strides=(2,2),
                padding='same',
                activation='relu'
                ),
        tf.keras.layers.Conv2D(
                filters=64, 
                kernel_size=(4,4),
                strides=(2,2),
                padding='same',
                activation='relu'
                ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32,
            activation='relu')
        ]

    def call(self, x):
        for _, layer in enumerate(self.layers): 
            x = layer(x)

        return x
        

class Deconv32(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Deconv32, self).__init__(**kwargs)
        self.layers = [
            tf.keras.layers.Dense(256,
                activation='relu'),
            tf.keras.layers.Reshape((2,2,64)),
            tf.keras.layers.Conv2DTranspose(
                filters=32, 
                kernel_size=(4,4), 
                strides=(2,2),
                padding='same',
                activation='relu',
                ),
            tf.keras.layers.Conv2DTranspose(
                filters=32, 
                kernel_size=(4,4),
                strides=(2,2),
                padding='same',
                activation='relu'
                ),
            tf.keras.layers.Conv2DTranspose(
                filters=64, 
                kernel_size=(4,4),
                strides=(2,2),
                padding='same',
                activation='relu'
                ),
            tf.keras.layers.Conv2DTranspose(
                filters=3, 
                kernel_size=(4,4),
                strides=(2,2),
                padding='same',
                activation='relu'
                ),
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=(4,4),
                padding='same',
                strides=(2,2)
                )
        ]
    def call(self, x):
        for _, layer in enumerate(self.layers): 
            x = layer(x)

        return x
