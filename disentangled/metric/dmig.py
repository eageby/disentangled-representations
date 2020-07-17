import disentangled.model.utils
import disentangled.utils as utils
import numpy as np
import sklearn.metrics
import tensorflow as tf

import gin


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[1]
    num_factors = ys.shape[1]
    m = np.zeros([num_codes, num_factors])

    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[:, j], mus[:, i])

    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)

    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[:, j], ys[:, j])

    return h


@gin.configurable('discretize', module='disentangled.metric.discrete_mutual_information_gap', blacklist=['target'])
def discretize(target, bins):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)

    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(
            target[i, :], np.histogram(target[i, :], bins)[1][:-1]
        )

    return discretized


@gin.configurable('discrete_mutual_information_gap', module='disentangled.metric')
def discrete_mutual_information_gap(model, dataset, points, batch_size, progress_bar=True):
    dataset = dataset.take(points).batch(batch_size, drop_remainder=True)
    mig = []

    if progress_bar:
        progress = disentangled.utils.TrainingProgress(dataset, total=points//batch_size)
        progress.write("Calculating Discrete MIG")
    else:
        progress = dataset

    for batch in progress:
        mean, log_var = model.encode(batch["image"])

        discrete_mean = discretize(mean, 20)
        mutual_information = discrete_mutual_info(discrete_mean, batch["label"])

        factor_entropy = discrete_entropy(batch["label"])
        mutual_information_sorted = tf.sort(mutual_information, axis=0, direction="DESCENDING")
        current_mig_score = (mutual_information_sorted[0, :] - mutual_information_sorted[1, :]) / factor_entropy

        mig.append(current_mig_score)

    return tf.reduce_mean(tf.stack(mig))
