import tensorflow as tf
import disentangled.utils
import disentangled.dataset
import disentangled.model
from disentangled.metric.factorvae import intact_dimensions, encode_dataset

def gini_index(representation):
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
            (
                representation
                / tf.norm(representation, 1, -1, True)
                * (N - k + 0.5)
                / N
            ),
            axis=-1,
        ),
        axis=0)

def metric(model, dataset, subset=100):
    if subset is not None:
        dataset = dataset.take(subset)

    breakpoint()
    encoded = encode_dataset(model, dataset)
    intact_idx = intact_dimensions(encoded, subset=None)
    breakpoint()

    progress = disentangled.utils.TrainingProgress(dataset, total=subset)
    gini = []
    for batch in progress:
        representation = model.encode(batch)[0]
        index = gini_index(representation)
        gini.append(index)
    
    batch_size = batch.shape[0]
    gini = tf.stack(gini, axis=0)
    return tf.reduce_mean(gini, axis=0)

if __name__ == '__main__':
    dataset = disentangled.dataset.shapes3d.pipeline()
    model = disentangled.model.utils.load("sparsevae_shapes3d")
    print(metric(model, dataset, 10))
