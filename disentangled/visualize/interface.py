import gin
import numpy as np
from . import show

@gin.configurable(module='disentangled.visualize')
def reconstructed(model, dataset, rows, cols):
    reconstructed, representation, target = model.predict(dataset, steps=1)
    show.grid(reconstructed, rows, cols)

@gin.configurable(module='disentangled.visualize')
def data(dataset, rows, cols):
    target = dataset.as_numpy_iterator().next()
    breakpoint()
    show.grid(target, rows, cols)

@gin.configurable(module='disentangled.visualize')
def fixed_factor_data(dataset, rows, cols, verbose):
    images = []
    for batch in dataset.take(rows).as_numpy_iterator():
        if verbose:
            print("Factor: {} = {}".format(batch['factor'], batch['factor_value']))
        images.append(batch['image'][:cols])
    
    show.grid(np.concatenate(images, axis=0), rows, cols)

@gin.configurable(module='disentangled.visualize')
def comparison(model, dataset, rows, cols):
    reconstructed, representation, target = model.predict(dataset, steps=1)
    show.comparison(target, reconstructed, rows, cols)
