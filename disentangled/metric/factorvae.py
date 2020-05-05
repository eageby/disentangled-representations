import disentangled.dataset as dataset
import disentangled.model.utils
import disentangled.utils as utils
import numpy as np
import tensorflow as tf
import scipy.stats

def majority_voting_classifier(data, n_latent, n_generative):
    V = np.zeros((n_latent, n_generative))

    for d, k in data:
        V[d, k] += 1

    import pdb;pdb.set_trace()
    return tf.math.argmax(V, axis=1)

    # for k in range(n_factors):
    #     v_matrix[j, k] = np.sum((d_values == j) & (k_values == k))


def encode_dataset(model, data):
    def encoding(element):
        representation, representations_variance = model.encode(element['image'])
        element['representation'] = representation
        element['representation_variance'] = representation_variance
        return element

    return data.map(encoding, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def representation_variance(data, subset=None):
    var = []

    if subset is not None:
        data = data.take(subset)
    progress = utils.TrainingProgress(data, total=subset)
    progress.write('Calculating Empirical Variance')
    for batch in progress:
        var.append(batch["representation"])

    return tf.math.reduce_variance(tf.concat(var, 0), axis=0) 

def intact_dimensions(data, subset=None, significance_level=0.05):
    representation_batches = []

    if subset is not None:
        data = data.take(subset)

    progress = utils.TrainingProgress(data, total=subset)
    progress.write('Calculating Collapsed Latent Dimensions')
    for batch in progress:
        representation_batches.append(batch['representation'])

    representations = np.concatenate(representation_batches)
    pvalues = np.array([scipy.stats.kstest(representations[i], 'norm').pvalue for i in range(representations.shape[-1])])
    idx =  np.where(pvalues < significance_level)[0]
    print('{} collapsed dimensions'.format(representations.shape[-1] - idx.size))
    return idx

def intact_dimensions_kld(data,subset=None, tolerance=1e-2):
    kld = []

    if subset is not None:
        data = data.take(subset)

    progress = utils.TrainingProgress(data, total=subset)
    progress.write('Calculating Collapsed Latent Dimensions')
    for batch in progress:
        mean = batch['representation'] 
        log_var = batch['representation_variance'] 
        kld.append(tf.reduce_mean(
            0.5 * ( tf.exp(log_var) - log_var + tf.square(mean) - 1)
            , axis=0))

    kld = tf.reduce_mean(tf.stack(kld, axis=0), axis=0)
    idx =  np.where(kld > tolerance)[0]
    print('{} collapsed dimensions'.format(kld.shape[-1] - len(idx)))
    return idx

def metric(model, dataset, training_votes=800, test_votes=500, subset=None):
    dataset = encode_dataset(model, dataset).take(training_votes)
    empirical_var = representation_variance(dataset, subset)
    # intact_idx = intact_dimensions(dataset, subset)
    intact_idx = intact_dimensions_kld(dataset, subset)

    samples = [] 
    progress = disentangled.utils.TrainingProgress(dataset, total=training_votes)
    progress.write('Calculating Metric Datapoints')
    for batch in progress:
        representations = batch['representation'] 

        representations /= tf.math.sqrt(empirical_var)
        representations_variance = tf.math.reduce_variance(
            representations, axis=0)
        dimension = tf.gather(intact_idx, tf.argmin(tf.gather(representations_variance, intact_idx)))

        samples.append(tf.stack((dimension, batch['factor'])))

    samples = tf.stack(samples)

    classifier = majority_voting_classifier(samples, representations.shape[-1], 6)

    dimensions = samples[:, 0]
    factors = samples[:, 1]

    estimate = tf.gather(classifier, dimensions)

    accuracy = np.sum(factors == estimate) / tf.size(dimensions, out_type=tf.int64)
    import pdb;pdb.set_trace()
    return accuracy
