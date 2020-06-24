import tensorflow as tf
import gin

gin.external_configurable(tf.keras.layers.Dense, "tf.keras.layers.Dense")
gin.external_configurable(tf.keras.layers.Conv2D, "tf.keras.layers.Conv2D")
gin.external_configurable(
    tf.keras.layers.Conv2DTranspose, "tf.keras.layers.Conv2DTranspose"
)

gin.external_configurable(tf.data.Dataset.shuffle, "tf.data.Dataset.shuffle")

gin.external_configurable(
    tf.keras.callbacks.TensorBoard, "tf.keras.callbacks.TensorBoard"
)

gin.constant("tf.data.experimental.AUTOTUNE", tf.data.experimental.AUTOTUNE)
