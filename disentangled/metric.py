import disentangled.dataset as dataset
import disentangled.model.utils
import disentangled.utils as utils
import numpy as np
import tensorflow as tf
import tqdm


def majority_voting_classifier(data, n_latent, n_generative):
    V = np.zeros((n_latent, n_generative))

    for d, k in data:
        V[d, k] += 1

        return V


def encode_labeled_dataset(model, data, batch_size=128):
    image_set = data.map(
        disentangled.dataset.utils.get_image).batch(batch_size)

    def interleaving(element):
        representation = model.encode(element)[0]

        return tf.data.Dataset.from_tensors({"representation": representation})

    representation_set = image_set.interleave(interleaving)
    data = tf.data.Dataset.zip((data.batch(batch_size), representation_set))

    def combine_sets(element1, element2):
        element1.update(element2)

        return element1

    return data.map(combine_sets).unbatch()

def fixed_factor_batches(data, batches, batch_size):
    for i in range(batches):
        factor = np.random.randint(0, len(dataset.shapes3d.factors))
        factor_value = np.random.randint(
            0, dataset.shapes3d.num_values_per_factor[dataset.shapes3d.factors[factor]]
        )
    

def representation_variance(data, batches, shuffle_buffer_size=100):
    data = data.shuffle(shuffle_buffer_size).take(batches)
    var = []

    for batch in tqdm.tqdm(data, total=batches):
        var.append(tf.math.reduce_variance(batch["representation"], axis=0))

    return tf.reduce_mean(tf.stack(var), axis=0)


if __name__ == "__main__":
    model = disentangled.model.utils.load("betavae_shapes3d")
    data = disentangled.dataset.shapes3d.as_image_label()

    data = encode_labeled_dataset(model, data)
    import pdb;pdb.set_trace()

    empirical_var = representation_variance(data, batches=10)

    samples = []

    for i in tqdm.tqdm(range(1000)):
        factor = np.random.randint(0, len(dataset.shapes3d.factors))
        factor_value = np.random.randint(
            0, dataset.shapes3d.num_values_per_factor[dataset.shapes3d.factors[factor]]
        )

        repr_map = lambda x: x['representation']
        for representations in data.take(1000).filter(fix_factor(factor, factor_value)).map(repr_map).take(1):

            import pdb;pdb.set_trace()
            representations /= tf.math.sqrt(empirical_var)
            representations_variance = tf.math.reduce_variance(
                representations, axis=0)
            dimension = tf.argmin(representations_variance)

            samples.append(tf.stack((dimension, factor)))

    samples = tf.stack(samples)

    train_data, test_data = tf.split(samples, 2, axis=0) 

    V = majority_voting_classifier(train_data, 32, 6)
    classifier = tf.math.argmax(V, axis=1)

    import pdb;pdb.set_trace()
    test_estimate = classifier[test_data[:, 0]]
    target = test_data[:, 1]

    error_rate = np.count_nonzero(target - test_estimate) / tf.size(target)

    print("Error Rate: {:%}".format(error_rate))
