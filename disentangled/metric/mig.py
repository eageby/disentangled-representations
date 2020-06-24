import disentangled.dataset as dataset
import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.utils
import disentangled.utils as utils
import numpy as np
import tensorflow as tf
from disentangled.model.objectives import _TOLERANCE

import sklearn


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])

    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])

    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)

    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])

    return h


def discretize(target, bins):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)

    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(
            target[i, :], np.histogram(target[i, :], bins)[1][:-1]
        )

    return discretized


def estimator(dataset):
    # network = networks.conv_4
    # breakpoint()
    # network.add(tf.keras.layers.Flatten())
    # network.add(tf.keras.layers.Dense(dataset.element_spec['label'].shape[-1], activation='softmax'))

    factor_possibilities = tf.cast(
        np.array([10.0, 10.0, 10.0, 8.0, 4.0, 15.0]), dtype=tf.float32
    )

    input_shape = tuple(dataset.element_spec["image"].shape[1:])

    inputs = tf.keras.Input(shape=input_shape)

    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights=None
    )
    mobilenet.trainable = True

    flatten = tf.keras.layers.Flatten()

    output_layers = [
        tf.keras.layers.Dense(int(n), activation="softmax", name=str(i))
        for i, n in enumerate(factor_possibilities)
    ]

    x = mobilenet(inputs)
    x = flatten(x)
    outputs = [out(x) for out in output_layers]

    estimator = tf.keras.Model(inputs=[inputs], outputs=outputs)

    dataset = dataset.map(
        lambda x: (
            x["image"],
            {str(i): l for i, l in enumerate(tf.unstack(x["label"], axis=-1))},
        )
    ).repeat()

    estimator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy() for i in outputs],
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),
    )
    estimator.fit(dataset, steps_per_epoch=50000)

    disentangled.model.utils.save(estimator, "mig_shapes")


def mutual_information_gap(model, dataset):
    factor_possibilities = tf.cast(
        np.array([10.0, 10.0, 10.0, 8.0, 4.0, 15.0]), dtype=tf.float32
    )

    factor_entropy = tf.math.log(factor_possibilities)
    dataset_size = tf.reduce_prod(factor_possibilities)

    log_prob_factors = -tf.math.log(factor_possibilities)
    log_px_condk = tf.reshape(
        tf.reduce_sum(log_prob_factors) - log_prob_factors, (1, 1, -1)
    )
    log_prob_factors = tf.reshape(log_prob_factors, (1, -1))

    mutual_information = []

    progress = disentangled.utils.TrainingProgress(dataset, total=480)

    for batch in progress:
        mean, log_var = model.encode(batch)
        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)
        samples = mean + tf.exp(0.5 * log_var) * noise

        batch_size = float(mean.shape[0])

        log_qzx_matrix = dist.Gaussian.log_likelihood(
            dist.Gaussian,
            samples,
            tf.expand_dims(mean, axis=1),
            tf.expand_dims(log_var, axis=0),
        )
        log_qz = tf.reduce_logsumexp(log_qzx_matrix, axis=1) - tf.math.log(
            batch_size * dataset_size
        )

        marginal_entropy = tf.reduce_mean(-log_qz, axis=0)

        log_qzx = dist.Gaussian.log_likelihood(
            dist.Gaussian,
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
    breakpoint()

    mig = tf.reduce_mean(
        normalized_mutual_information[0, :] - normalized_mutual_information[1, :]
    )

    return tf.reduce_mean(tf.stack(mig))


def mutual_information_gap_slow(model, dataset):
    factor_possibilities = tf.cast(
        np.array([10.0, 10.0, 10.0, 8.0, 4.0, 15.0]), dtype=tf.float32
    )

    factor_entropy = tf.math.log(factor_possibilities)
    dataset_size = tf.reduce_prod(factor_possibilities)

    mutual_information = []
    progress = disentangled.utils.TrainingProgress(dataset, total=480)

    for batch in progress:
        mean, log_var = model.encode(batch["image"])
        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)
        samples = mean + tf.exp(0.5 * log_var) * noise

        batch_size = float(mean.shape[0])

        log_qzx_matrix = dist.Gaussian.log_likelihood(
            dist.Gaussian,
            samples,
            tf.expand_dims(mean, axis=1),
            tf.expand_dims(log_var, axis=0),
        )
        log_qz = tf.reduce_logsumexp(log_qzx_matrix, axis=1) - tf.math.log(
            batch_size * dataset_size
        )

        marginal_entropy = tf.reduce_mean(-log_qz, axis=0)

        log_qzx = dist.Gaussian.log_likelihood(
            dist.Gaussian,
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
    breakpoint()

    mig = tf.reduce_mean(
        normalized_mutual_information[0, :] - normalized_mutual_information[1, :]
    )

    return tf.reduce_mean(tf.stack(mig))


if __name__ == "__main__":
    dataset = disentangled.dataset.shapes3d.pipeline(batch_size=1000).take(100)
    model = disentangled.model.utils.load("factorvae_shapes3d")

    # estimator(dataset)
    print(mutual_information_gap(model, dataset))
