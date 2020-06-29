import disentangled.dataset.utils 
import disentangled.model.utils
import disentangled.utils as utils
import numpy as np
import tensorflow as tf
import scipy.stats
import disentangled.model.distributions as dist
import gin

from disentangled.metric.utils import intact_dimensions_kld


def majority_voting_classifier(data, n_latent, n_generative):
    V = np.zeros((n_latent, n_generative))

    dimension, factor = tf.split(data, 2, axis=-1)
    for i in range(n_latent):
        for j in range(n_generative):
            V[i, j] = np.sum((dimension == i) & (factor == j))

    return np.argmax(V, axis=1)

    # for k in range(n_factors):
    #     v_matrix[j, k] = np.sum((d_values == j) & (k_values == k))


# def encode_dataset(model, data):
#     def encoding(element):
#         mean, log_var = model.encode(element['image'])
#         # representation = model.sample(mean, variance, training=True)
#         noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)
#         representation = mean + tf.exp(0.5 * log_var) * noise

#         element['representation'] = representation
#         element['representation_mean'] = mean
#         element['representation_log_var'] = log_var
#         return element

#     return data.map(encoding, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def representation_variance(model, data, subset=None, progress_bar=True):
    var = []

    if subset is not None:
        data = data.take(subset)

    if progress_bar:
        progress = utils.TrainingProgress(data, total=subset)
        progress.write("Calculating Empirical Variance")
    else:
        progress = data

    for batch in progress:
        var.append(tf.math.reduce_variance(model.encode(batch["image"])[0], axis=0))

    return tf.math.reduce_mean(tf.stack(var, 0), axis=0)


# def intact_dimensions(data, subset=None, significance_level=0.05):
#     representation_batches = []

#     if subset is not None:
#         data = data.take(subset)

#     progress = utils.TrainingProgress(data, total=subset)
#     progress.write('Calculating Collapsed Latent Dimensions')
#     for batch in progress:
#         representation_batches.append(batch['representation'])

#     representations = np.concatenate(representation_batches)
#     pvalues = np.array([scipy.stats.kstest(representations[i], 'norm').pvalue for i in range(representations.shape[-1])])
#     idx =  np.where(pvalues < significance_level)[0]
#     print('{} collapsed dimensions'.format(representations.shape[-1] - idx.size))
#     return idx





@gin.configurable(module='disentangled.metric')
def factorvae_score(
    model, dataset, training_votes, subset, prior_dist, tolerance, progress_bar=True
):
    # dataset = encode_dataset(model, dataset)
    empirical_var = representation_variance(model, dataset, subset=subset, progress_bar=progress_bar)
    # intact_idx = intact_dimensions(dataset, subset)
    image_dataset = dataset.map(disentangled.dataset.utils.get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    intact_idx = intact_dimensions_kld(model, dataset, prior_dist=prior_dist, subset=subset, tolerance=tolerance, progress_bar=progress_bar)

    if tf.size(intact_idx) == 0:
        return 0.0

    dataset = dataset.take(training_votes)

    if progress_bar:
        progress = disentangled.utils.TrainingProgress(
            dataset.take(training_votes), total=training_votes
        )
        progress.write("Calculating Metric Datapoints")
    else:
        progress = dataset

    samples = []
    for batch in progress:
        representations = model.encode(batch["image"])[0]

        representations /= tf.math.sqrt(empirical_var)
        representations_variance = tf.math.reduce_variance(representations, axis=0)
        dimension = tf.argmin(tf.gather(representations_variance, intact_idx))

        samples.append(tf.stack((dimension, batch["factor"])))

    samples = tf.stack(samples)

    classifier = majority_voting_classifier(samples, intact_idx.size, 6)

    dimensions = samples[:, 0]
    factors = samples[:, 1]

    estimated_factors = classifier[np.array(dimensions)]

    accuracy = np.sum(factors == estimated_factors) / tf.size(
        factors, out_type=tf.int64
    )
    return accuracy
