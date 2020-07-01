import disentangled.dataset.utils
import disentangled.metric.utils
import disentangled.model.utils
import disentangled.utils 
import gin
import numpy as np
import tensorflow as tf


def _majority_voting_classifier(data, n_latent, n_generative):
    V = np.zeros((n_latent, n_generative))

    dimension, factor = tf.split(data, 2, axis=-1)

    for i in range(n_latent):
        for j in range(n_generative):
            V[i, j] = np.sum((dimension == i) & (factor == j))

    return np.argmax(V, axis=1)


@gin.configurable(module="disentangled.metric")
def factorvae_score(
    model, dataset, training_votes, test_votes, tolerance, progress_bar=True
):
    empirical_var = disentangled.metric.utils.representation_variance(
        model, dataset)
    intact_idx = np.where(empirical_var > tolerance)[0]

    fixed_factor_set = disentangled.metric.utils.fixed_factor_dataset(
        dataset, batch_size=gin.REQUIRED, num_values_per_factor=gin.REQUIRED
    ).take(training_votes + test_votes)

    if progress_bar:
        fixed_factor_set = disentangled.utils.TrainingProgress(
            fixed_factor_set, total=training_votes + test_votes
        )
        fixed_factor_set.write("{} intact dimensions".format(intact_idx.size))
        fixed_factor_set.write("Calculating Metric Datapoints")

    if tf.size(intact_idx) == 0:
        return 0.0

    samples = []

    for batch in fixed_factor_set:
        representations = model.encode(batch["image"])[0]

        representations /= tf.math.sqrt(empirical_var)
        representations_variance = tf.math.reduce_variance(
            representations, axis=0)
        dimension = tf.argmin(tf.gather(representations_variance, intact_idx))

        samples.append(
            tf.stack((dimension, tf.cast(batch["factor"], tf.int64))))

    samples = tf.stack(samples)
    train, test = tf.split(samples, [training_votes, test_votes])

    classifier = _majority_voting_classifier(train, intact_idx.size, 6)

    dimensions = test[:, 0]
    factors = test[:, 1]

    estimated_factors = classifier[np.array(dimensions)]

    accuracy = np.sum(factors == estimated_factors) / tf.size(
        factors, out_type=tf.int64
    )

    return accuracy
