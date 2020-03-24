import sys
from inspect import signature
import tensorflow as tf 
import disentangled.dataset
import disentangled.model
import disentangled.visualize

__all__ = ["train"]

def filter_parameters(parameters, function):
    parameters = {k:v for k, v in parameters.items() if k in signature(function).parameters}
    return parameters

def train(model_constructor, dataset, batch_size, **kwargs) -> tf.keras.Model:
    tf.random.set_seed(10)
    
    model_parameters = filter_parameters(kwargs, model_constructor)
    model = model_constructor(**model_parameters)

    data = dataset.pipeline(batch_size=batch_size)

    model.predict(data, steps=1) # Instantiating model 
    model.train(data.repeat(), **kwargs)
    
    return model
