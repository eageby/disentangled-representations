import disentangled.dataset as dataset
import disentangled.model.utils
import disentangled.visualize as vi
import gin
import numpy as np
import tensorflow as tf


@gin.configurable(module="disentangled.visualize")
def traversal2d(model, data, sample, skip_batches, steps, offset, shuffle_fn=None):
    if shuffle_fn is not None:
        data = shuffle_fn(data)

    data = data.skip(skip_batches)
    batch = data.as_numpy_iterator().next()

    representations = model.encode(batch)[0]

    min_values = tf.math.reduce_min(representations, axis=0)
    max_values = tf.math.reduce_max(representations, axis=0)

    variances = tf.math.reduce_variance(representations, axis=0)
    dimensions = np.argsort(variances)[:-3:-1]

    z = np.array(representations[sample])

    range_ = [
        np.linspace(min_values[d] - offset, max_values[d] + offset, num=steps)

        for d in dimensions
    ]
    grid = np.meshgrid(*range_)

    grid = np.stack([np.reshape(g, -1) for g in grid], axis=1)
    z_traversal = np.broadcast_to(z, (steps ** 2, z.shape[-1])).copy()
    z_traversal[:, dimensions] = grid

    x_traversal = model.decode(z_traversal)[0]

    vi.show.show_grid(
        vi.show.batch_to_grid(x_traversal, steps, steps),
        title="Latent Space Traversal 2D",
    )


@gin.configurable(module="disentangled.visualize")
def traversal1d(
    model, data, skip_batches, sample, dimensions, steps, offset, shuffle_fn=None
):
    if shuffle_fn is not None:
        data = shuffle_fn(data)

    data = data.skip(skip_batches)
    batch = data.as_numpy_iterator().next()
    representations = model.encode(batch)[0]

    variances = tf.math.reduce_variance(representations, axis=0)
    dimensions = np.argsort(variances)[: -(dimensions + 1): -1]

    min_values = tf.math.reduce_min(representations, axis=0)
    max_values = tf.math.reduce_max(representations, axis=0)

    z = np.array(representations[sample])

    range_ = np.stack(
        [
            np.linspace(min_values[d] - offset,
                        max_values[d] + offset, num=steps)

            for d in dimensions
        ],
        axis=0,
    )

    z_traversal = np.broadcast_to(
        z, (dimensions.size, steps, z.shape[-1])).copy()
    idx = np.arange(dimensions.size)

    z_traversal[idx, :, dimensions] = range_

    z_traversal = np.reshape(z_traversal, (-1, z.shape[-1]))
    x_traversal = model.decode(z_traversal)[0]

    vi.show.show_grid(
        vi.show.batch_to_grid(x_traversal, rows=dimensions.size, cols=steps),
        title="Latent Space Traversal 1D",
    )
