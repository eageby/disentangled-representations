import disentangled.dataset as dataset
import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.utils
import disentangled.utils as utils
import numpy as np
import tensorflow as tf
import gin

def discrete_entropy(data):
    tolerance = 1e-7
    n_factors = data.shape[1]
    entropy = np.zeros((n_factors,))

    for i in range(n_factors):
        occurences = np.bincount(data[:, i])
        prob = occurences / np.sum(occurences)

        breakpoint()
        entropy[i] = np.sum(prob * np.log(prob + tolerance))

    return entropy   

@gin.configurable('mutual_information_gap', module='disentangled.metric')
def mutual_information_gap(model, dataset, batches, batch_size, progress_bar=True):
    dataset = dataset.batch(batch_size).take(batches)

    mutual_information = []
    if progress_bar:
        progress = disentangled.utils.TrainingProgress(dataset, total=batches)
        progress.write("Calculating MIG")
    else:
        progress = dataset

    for batch in progress:
        mean, log_var = model.encode(batch['image'])
        samples = model.sample(mean, log_var, training=True)

        factor_entropy = discrete_entropy(batch['label'])
        log_qzx_matrix = prior_dist.log_likelihood(
            samples,
            tf.expand_dims(mean, axis=1),
            tf.expand_dims(log_var, axis=0),
        )
        log_qz = tf.reduce_logsumexp(log_qzx_matrix, axis=1) - tf.math.log(
            batch_size * dataset_size
        )

        marginal_entropy = tf.reduce_mean(-log_qz, axis=0)

        log_qzx = prior_dist(
            tf.expand_dims(samples, -1),
            tf.expand_dims(mean, -1),
            tf.expand_dims(log_var, -1),
        )
        conditional_entropy = -tf.reduce_mean(
            tf.expand_dims(log_qzx, -1) + log_px_condk, axis=0
        )
        mutual_information.append(
            tf.expand_dims(marginal_entropy, -1) - conditional_entropy
        )

    mutual_information = tf.reduce_mean(tf.stack(mutual_information, 0), 0)
    mutual_information = tf.sort(mutual_information, axis=0, direction="DESCENDING")

    normalized_mutual_information = mutual_information / factor_entropy

    mig = tf.reduce_mean(
        normalized_mutual_information[0, :] - normalized_mutual_information[1, :]
    )

    return tf.reduce_mean(tf.stack(mig))
