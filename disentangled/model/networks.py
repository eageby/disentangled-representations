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

gin.constant('networks.conv_2', conv_2)
gin.constant('networks.conv_2_transpose', conv_2_transpose)
gin.constant('networks.conv_4', conv_4)
gin.constant('networks.conv_4_transpose', conv_4_transpose)
