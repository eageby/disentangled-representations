import sys
import tensorflow as tf 
import disentangled.dataset
import disentangled.model
import disentangled.visualize

__all__ = ["train"]


def train(model: tf.keras.Model, dataset: tf.data.Dataset, hyperparameters: dict) -> tf.keras.Model:
    tf.random.set_seed(10)
    
    data = dataset.pipeline(batch_size=hyperparameters['batch_size'])

    optimizer = tf.keras.optimizers.get(hyperparameters['optimizer'])
    model.compile(optimizer)
    model.fit(data.repeat(), steps_per_epoch=hyperparameters['iterations'])
    
    return model
