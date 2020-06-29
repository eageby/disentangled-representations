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


def discretize(target, bins):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)

    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(
            target[i, :], np.histogram(target[i, :], bins)[1][:-1]
        )

    return discretized


@gin.configurable('mutual_information_gap', module='disentangled.metric')
def mutual_information_gap(model, dataset, batches, batch_size, progress_bar=True):
    dataset = dataset.batch(batch_size).take(batches)
    mig = []

    if progress_bar:
        progress = disentangled.utils.TrainingProgress(dataset, total=batches)
        progress.write("Calculating MIG")
    else:
        progress = dataset

    for batch in progress:
        mean, _ = model.encode(batch["image"])

        discrete_mean = discretize(mean, 20)
        mutual_information = discrete_mutual_info(discrete_mean, batch["label"])

        factor_entropy = discrete_entropy(batch["label"])
        mutual_information_sorted = np.sort(mutual_information, axis=0)[::-1]

        mig.append(
            (mutual_information_sorted[0, :] - mutual_information_sorted[1, :])
            / factor_entropy
        )

    return tf.reduce_mean(tf.stack(mig))

if __name__ == "__main__":
    disentangled.utils.parse_config_file("mig.gin")
    print(
        mutual_information_gap(
            model=gin.REQUIRED,
            dataset=gin.REQUIRED,
            batches=gin.REQUIRED,
            batch_size=gin.REQUIRED,
        )
    )
