import sys
from inspect import signature
import tensorflow as tf 
import disentangled.dataset
import disentangled.model
import disentangled.visualize
import gin.tf

__all__ = ["train"]

@gin.configurable
def train(model, dataset, batch_size) -> tf.keras.Model:
    data = dataset.pipeline(batch_size=batch_size)

    model.predict(data, steps=1) # Instantiating model 
    model.train(data.repeat())
    
    return model

if __name__ == '__main__':
    gin.parse_config_file('disentangled/config/betavae/shapes3d.gin')
    train(batch_size=1)
    print(gin.operative_config_str())
