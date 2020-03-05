import tensorflow as tf
import numpy as np

import disentangled.model.utils

import disentangled.dataset as dataset 
import disentangled.visualize as vi

def traversal_2d(model, data, steps=20):
    batch = data.pipeline().as_numpy_iterator().next()
    representations = model.encode(batch)[0]

    min_values = tf.math.reduce_min(representations, axis=0)
    max_values = tf.math.reduce_max(representations, axis=0)
    
    variances = tf.math.reduce_variance(representations, axis=0)
    dimensions = np.argsort(variances)[:-3:-1]

    z = np.array(representations[0])

    range_ = [np.linspace(min_values[d], max_values[d], num=steps) for d in dimensions]
    grid = np.meshgrid(*range_)

    grid = np.stack([np.reshape(g, -1) for g in grid], axis=1)
    z_traversal = np.broadcast_to(z, (steps**2, z.shape[-1])).copy()
    z_traversal[:, dimensions] = grid
    
    x_traversal = model.decode(z_traversal)[0]
    
    vi.show_grid(vi.batch_to_grid(x_traversal, steps, steps))

def traversal_1d(model,data,  dimensions=10, steps=31):
    batch = data.pipeline().as_numpy_iterator().next()
    representations = model.encode(batch)[0]

    variances = tf.math.reduce_variance(representations, axis=0)
    dimensions = np.argsort(variances)[:-(dimensions + 1) :-1]

    min_values = tf.math.reduce_min(representations, axis=0)
    max_values = tf.math.reduce_max(representations, axis=0)
    
    z = np.array(representations[6])

    range_ = np.stack([np.linspace(min_values[d], max_values[d], num=steps) for d in dimensions], axis=0)

    z_traversal = np.broadcast_to(z, (dimensions.size, steps, z.shape[-1])).copy()
    idx = np.arange(dimensions.size)

    z_traversal[idx, :, dimensions] = range_
    
    z_traversal = np.reshape(z_traversal, (-1, z.shape[-1]))
    x_traversal = model.decode(z_traversal)[0]

    vi.show_grid(vi.batch_to_grid(x_traversal, rows=dimensions.size, cols=steps))
