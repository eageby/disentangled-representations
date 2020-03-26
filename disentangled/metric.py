import disentangled.dataset as dataset
import disentangled.model.utils
import disentangled.utils as utils
import numpy as np
import tensorflow as tf


def majority_voting_classifier(data, n_latent, n_generative):
    V = np.zeros((n_latent, n_generative))

    for d, k in data:
        V[d, k] += 1

        return tf.math.argmax(V, axis=1)

def encode_dataset(model, data):
    def encoding(element):
        representation = model.encode(element['image'])[0]
        element['representation'] = representation
        return element

    return data.map(encoding, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def representation_variance(data):
    var = []

    for batch in data:
        var.append(tf.math.reduce_variance(batch["representation"], axis=0))

    return tf.reduce_mean(tf.stack(var), axis=0)


def metric_factorvae(model, dataset):
    dataset = encode_dataset(model, dataset).take(130)
    empirical_var = representation_variance(dataset)

    samples = [] 
    for batch in dataset:
        representations = batch['representation'] 
        representations /= tf.math.sqrt(empirical_var)
        representations_variance = tf.math.reduce_variance(
            representations, axis=0)
        dimension = tf.argmin(representations_variance)

        samples.append(tf.stack((dimension, batch['factor'])))

    samples = tf.stack(samples)

    train_data, test_data = tf.split(samples, [50, 80], axis=0) 

    classifier = majority_voting_classifier(train_data, 32, 6)

    test_estimate = tf.gather(classifier, test_data[:, 0])
    target = test_data[:, 1]

    error_rate = np.count_nonzero(target - test_estimate) / tf.size(target)
    return error_rate

if __name__ == "__main__":
    model = disentangled.model.utils.load("betavae_shapes3d")
    import timeit
    dataset = dataset.datasets.Shapes3d_ordered.create(batch_size=10)
    error_rate = metric_factorvae(model, dataset)
    print("Error Rate: {:%}".format(error_rate))
