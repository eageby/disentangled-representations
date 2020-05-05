import disentangled.dataset as dataset
import disentangled.model.utils
import disentangled.utils as utils
import numpy as np
import tensorflow as tf
from disentangled.model.objectives import logsumexp, _TOLERANCE


def entropy(mean, log_var):
    log_likelihood = log_likelihood_gaussian(mean, log_var)

    return tf.reduce_mean(-log_likelihood, axis=0)

def normalized_log_likelihood_gaussian(mean, log_var):
    return -0.5*(tf.square(mean) / (tf.exp(log_var) + _TOLERANCE))
 
def mutual_information_gap(model, dataset):
    factor_possibilities = tf.cast(
        np.array([10.0, 10.0, 10.0, 8.0, 4.0, 15.0]), dtype=tf.float32
    )

    log_prob_factors = -tf.math.log(factor_possibilities)
    factor_entropy = tf.math.log(factor_possibilities)
    log_px_condk = tf.reduce_sum(log_prob_factors) - log_prob_factors

    mig = []

    for batch in dataset:
        z_mean, z_log_var = model.encode(batch["image"])
        log_qzx = normalized_log_likelihood_gaussian(z_mean, z_log_var)

        marginal_entropy = -tf.expand_dims(
                    tf.reduce_mean(log_qzx, axis=0),
                axis=-1)

        conditional_entropy = -logsumexp(
            tf.expand_dims(log_qzx, axis=-1) +
            tf.reshape(log_prob_factors, (1, 1, -1)),
            axis=0,
        )

        mutual_information = marginal_entropy - conditional_entropy
        mutual_information = tf.sort(
            mutual_information, axis=0, direction="DESCENDING")

        normalized_mutual_information = mutual_information / factor_entropy

        mig.append(
            tf.reduce_mean(
                (normalized_mutual_information[0, :] -
                 normalized_mutual_information[1, :]) 
            )
        )

    import pdb;pdb.set_trace()
    return tf.reduce_mean(tf.stack(mig))


if __name__ == "__main__":
    disentangled.utils.disable_gpu()
    dataset = disentangled.dataset.shapes3d.as_image_label().batch(128).take(10)
    model = disentangled.model.utils.load("betavae_shapes3d")
    print(mutual_information_gap(model, dataset))
