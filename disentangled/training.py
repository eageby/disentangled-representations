import sys
import tensorflow as tf 
import disentangled.dataset
import disentangled.model
import disentangled.visualize

__all__ = ["train"]


def train(model: tf.keras.Model, dataset: tf.data.Dataset, batch_size=128, learning_rate=1e-3, iterations=100) -> tf.keras.Model:
    tf.random.set_seed(10)
    
    data = dataset.pipeline(batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer)
    model.fit(data.repeat(), steps_per_epoch=iterations)
    
    return model
