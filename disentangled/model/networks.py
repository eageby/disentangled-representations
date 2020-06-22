import tensorflow as tf
import gin

conv_4 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Conv2D(
            32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Conv2D(
            64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Conv2D(
            64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
        ),
    ])

conv_4_transpose = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(
                32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
        ])

conv_2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Conv2D(
            64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
        ),
    ])

conv_2_transpose = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(
                32, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                64, kernel_size=(4, 4), strides=(2, 2), activation="relu"
            ),
        ])

@gin.configurable
def discriminator(latents, activation):
    return tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        1000, activation=activation, input_shape=(latents,)
                    ),
                    tf.keras.layers.Dense(1000, activation=activation),
                    tf.keras.layers.Dense(1000, activation=activation),
                    tf.keras.layers.Dense(1000, activation=activation),
                    tf.keras.layers.Dense(1000, activation=activation),
                    tf.keras.layers.Dense(1000, activation=activation),
                    tf.keras.layers.Dense(2, activation="softmax"),
                ]
            )

gin.constant('disentangled.model.networks.conv_2', conv_2)
gin.constant('disentangled.model.networks.conv_2_transpose', conv_2_transpose)
gin.constant('disentangled.model.networks.conv_4', conv_4)
gin.constant('disentangled.model.networks.conv_4_transpose', conv_4_transpose)
