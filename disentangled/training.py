import sys
from inspect import signature
import tensorflow as tf 
import disentangled.dataset
import disentangled.model
import disentangled.visualize

__all__ = ["train"]

def train(model_constructor, dataset, batch_size, **kwargs) -> tf.keras.Model:
    model = model_constructor(**kwargs)
    data = dataset.pipeline(batch_size=batch_size)

    model.predict(data, steps=1) # Instantiating model 
    model.train(data.repeat(), **kwargs)
    
    return model
