import tensorflow as tf

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
