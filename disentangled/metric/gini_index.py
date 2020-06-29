import disentangled.dataset
import disentangled.model
import disentangled.utils
import gin
import tensorflow as tf
from disentangled.metric.utils import intact_dimensions_kld


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
def gini_index(model, dataset, prior_dist, tolerance, subset, progress_bar=True):
    if subset is not None:
        dataset = dataset.take(subset)

    intact_idx = intact_dimensions_kld(model, dataset, tolerance, prior_dist, subset=None)

    if progress_bar:
        progress = disentangled.utils.TrainingProgress(dataset, total=subset)
        progress.write('Calculating Gini Index')
    else:
        progress = dataset

    gini = []

    for batch in progress:
        representation = model.encode(batch)[0]
        index = _gini_index_representation(
            tf.gather(representation, intact_idx, axis=1)
        )
        gini.append(index)

    gini = tf.stack(gini, axis=0)

    return tf.reduce_mean(gini, axis=0)
