import disentangled.dataset
import disentangled.model
import disentangled.utils
import disentangled.metric.utils
import numpy as np
import gin
import tensorflow as tf


def _gini_index_representation(representation):
    """Returns the gini_index of a representation, being the sparsity measure.
    1 is the most sparse, 0 is the least.

    Args: 
        representation (tf.Tensor): Batch axis 0 and feature axis 1
    Returns:
        (float): gini_index
    """
    representation = tf.sort(tf.math.abs(representation))
    idx = tf.where(tf.not_equal(tf.norm(representation, 1, -1), 0.0))
    representation = tf.gather_nd(representation, idx)

    N = representation.shape[-1]
    k = tf.reshape(1.0 + tf.range(N, dtype=tf.float32), [1, N])

    return tf.reduce_mean(
        1.0
        - 2.0
        * tf.reduce_sum(
            (representation / tf.norm(representation, 1, -1, True) * (N - k + 0.5) / N),
            axis=-1,
        ),
        axis=0,
    )


@gin.configurable(module="disentangled.metric")
def gini_index(model, dataset, samples, batch_size, tolerance, progress_bar=True):
    empirical_var = disentangled.metric.utils.representation_variance(
        model, dataset)
    intact_idx = np.where(empirical_var > tolerance)[0]

    dataset = dataset.batch(batch_size).take(samples)
    if progress_bar:
        progress = disentangled.utils.TrainingProgress(dataset, total=samples)
        progress.write('Calculating Gini Index')
    else:
        progress = dataset

    all_index = None
    for batch in progress:
        representation = model.encode(batch['image'])[0]
        index = _gini_index_representation(
            tf.gather(representation, intact_idx, axis=1)
        )
        index = tf.expand_dims(index, axis=0)

        if all_index is None:
            all_index = index
        else:
            all_index = tf.concat([all_index, index], axis=0)

    return tf.reduce_mean(all_index, axis=0)
