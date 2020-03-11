import sys
import tensorflow as tf 
import disentangled.dataset
import disentangled.model
import disentangled.visualize

__all__ = ["train"]


def train(model_constructor, dataset: tf.data.Dataset, batch_size, learning_rate, iterations, **model_parameters) -> tf.keras.Model:
    tf.random.set_seed(10)
    
    model = model_constructor(**model_parameters)

    data = dataset.pipeline(batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer)
    model.fit(data.repeat(), steps_per_epoch=iterations)
    
    return model
