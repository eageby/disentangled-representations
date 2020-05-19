import disentangled.dataset as dataset 
import disentangled.model.utils
import disentangled.utils as utils
import disentangled.model.distributions as dist
import numpy as np
import tensorflow as tf
from disentangled.model.objectives import _TOLERANCE


def mutual_information_gap(model, dataset):
    factor_possibilities = tf.cast(
        np.array([10.0, 10.0, 10.0, 8.0, 4.0, 15.0]), dtype=tf.float32
    )

    factor_entropy = tf.math.log(factor_possibilities)
    dataset_size = tf.reduce_prod(factor_possibilities)
    
    log_prob_factors = -tf.math.log(factor_possibilities)
    log_px_condk = tf.reshape(tf.reduce_sum(log_prob_factors) - log_prob_factors, (1,1,-1))
    log_prob_factors = tf.reshape(log_prob_factors, (1, -1))

    mutual_information = []

    progress = disentangled.utils.TrainingProgress(dataset, total=480)
    for batch in progress:
        mean, log_var = model.encode(batch["image"])
        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)
        samples = mean + tf.exp(0.5 * log_var) * noise

        batch_size = float(mean.shape[0])

        log_qzx_matrix = dist.Gaussian.log_likelihood(
                dist.Gaussian, samples, tf.expand_dims(mean, axis=1), tf.expand_dims(log_var, axis=0)
        )
        log_qz = tf.reduce_logsumexp(log_qzx_matrix, axis=1) - tf.math.log(batch_size * dataset_size)
        
        marginal_entropy = tf.reduce_mean(-log_qz, axis=0)
        
        log_qzx = dist.Gaussian.log_likelihood(
                dist.Gaussian, tf.expand_dims(samples, -1), tf.expand_dims(mean,-1), tf.expand_dims(log_var,-1)
        )
        conditional_entropy = - tf.reduce_mean(tf.expand_dims(log_qzx, -1) + log_px_condk, axis=0)
        mutual_information.append(tf.expand_dims(marginal_entropy, -1) - conditional_entropy)

    mutual_information = tf.reduce_mean(tf.stack(mutual_information, 0), 0)
    mutual_information = tf.sort(
        mutual_information, axis=0, direction="DESCENDING")

    normalized_mutual_information = mutual_information / factor_entropy
    breakpoint()

    mig = tf.reduce_mean(
        normalized_mutual_information[0, :] -
        normalized_mutual_information[1, :]
    )

    return tf.reduce_mean(tf.stack(mig))


if __name__ == "__main__":
    dataset = disentangled.dataset.shapes3d.as_image_label().batch(1000)
    model = disentangled.model.utils.load("factorvae_shapes3d")
    print(mutual_information_gap(model, dataset))
