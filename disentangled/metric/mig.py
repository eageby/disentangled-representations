import disentangled.dataset as dataset
import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.utils
import disentangled.utils as utils
import numpy as np
import tensorflow as tf
import gin

def estimator(dataset):
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



@gin.configurable('mutual_information_gap', module='disentangled.metric')
def mutual_information_gap(model, dataset, batches, batch_size, prior_dist, progress_bar=True):
    dataset = dataset.batch(batch_size).take(batches)

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

    if progress_bar:
        progress = disentangled.utils.TrainingProgress(dataset, total=batches)
        progress.write("Calculating MIG")
    else:
        progress = dataset

    for batch in progress:
        mean, log_var = model.encode(batch)
        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)
        samples = mean + tf.exp(0.5 * log_var) * noise

        batch_size = float(mean.shape[0])

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
    breakpoint()

    mig = tf.reduce_mean(
        normalized_mutual_information[0, :] - normalized_mutual_information[1, :]
    )

    return tf.reduce_mean(tf.stack(mig))
