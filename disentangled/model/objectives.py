import math

import disentangled.model.distributions as dist
import gin
import tensorflow as tf

@gin.configurable(whitelist=["objective_fn"])
class Objective(tf.keras.layers.Layer):
    def __init__(self, objective_fn, **kwargs):
        super(Objective, self).__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten()
        self.objective_fn = objective_fn

    def call(self, *args):
        return self.objective_fn(self, *[self.flatten(i) for i in args])


@gin.configurable(whitelist=["beta", "prior_dist", "output_dist"], module="objectives")
@tf.function
def betavae(
        objective,
        target,
        x_mean,
        x_log_var,
        z,
        z_mean,
        z_log_var,
        beta,
        prior_dist,
        output_dist,
):
    log_likelihood = tf.reduce_mean(
        tf.reduce_sum(output_dist.log_likelihood(
            target, x_mean, x_log_var), axis=1),
        axis=0,
    )

    kld = tf.reduce_mean(
        tf.reduce_sum(prior_dist.kld(z_mean, z_log_var), axis=1), axis=0
    )

    objective.add_metric(-log_likelihood,
                         aggregation="mean", name="-loglikelihood")
    objective.add_metric(kld, aggregation="mean", name="kld")

    return -log_likelihood + beta * kld


@gin.configurable(
    whitelist=["gamma", "prior_dist", "output_dist", "tolerance"], module="objectives"
)
@tf.function
def factorvae(
        objective,
        target,
        x_mean,
        x_log_var,
        z_mean,
        z_log_var,
        discriminator_probability,
        gamma,
        prior_dist,
        output_dist,
        tolerance,
):

    log_likelihood = tf.reduce_mean(
        tf.reduce_sum(output_dist.log_likelihood(
            target, x_mean, x_log_var), axis=1),
        axis=0,
    )

    kld = tf.reduce_mean(
        tf.reduce_sum(prior_dist.kld(z_mean, z_log_var), axis=1), axis=0
    )

    kld_discriminator = tf.reduce_mean(
        tf.math.log(
            discriminator_probability /
            ((1 - discriminator_probability) + tolerance)
        )
    )

    objective.add_metric(-log_likelihood,
                         aggregation="mean", name="-loglikelihood")
    objective.add_metric(kld, aggregation="mean", name="kld1")
    objective.add_metric(kld_discriminator, aggregation="mean", name="kld2")

    return -log_likelihood + kld + gamma * kld_discriminator


@gin.configurable(module="objectives", whitelist=["tolerance"])
def discriminator_loss(p_z, p_permuted, tolerance):
    return -tf.reduce_mean(
        tf.math.log(p_z + tolerance) + tf.math.log(p_permuted + tolerance)
    )


@tf.function
@gin.configurable(
    whitelist=["beta", "dataset_size", "prior_dist", "output_dist"], module="objectives"
)
def betatcvae(
        objective,
        target,
        x_mean,
        x_log_var,
        z,
        z_mean,
        z_log_var,
        prior_dist,
        output_dist,
        beta,
        dataset_size,
):
    log_nm = tf.math.log(float(target.shape[0] * dataset_size))

    log_px = tf.reduce_sum(
        output_dist.log_likelihood(target, x_mean, x_log_var), axis=1)
    log_pz = tf.reduce_sum(prior_dist.log_likelihood(z), axis=1)

    log_qzx = tf.reduce_sum(
        prior_dist.log_likelihood(z, z_mean, z_log_var), axis=1)

    log_qz_matrix = prior_dist.log_likelihood(
        z, tf.expand_dims(z_mean, axis=1), tf.expand_dims(z_log_var, axis=0)
    )

    log_qz_prod = tf.reduce_sum(
        tf.reduce_logsumexp(log_qz_matrix, axis=1) - log_nm, axis=1
    )

    log_qz = tf.reduce_logsumexp(tf.reduce_sum(
        log_qz_matrix, axis=2), axis=1) - log_nm

    mutual_information = log_qzx - log_qz
    total_correlation = log_qz - log_qz_prod
    kld = log_qz_prod - log_pz

    objective.add_metric(-log_px, aggregation="mean", name="-loglikelihood")
    objective.add_metric(mutual_information, aggregation="mean", name="info")
    objective.add_metric(kld, aggregation="mean", name="kld")
    objective.add_metric(total_correlation, aggregation="mean", name="tc")

    return -tf.reduce_mean(
        log_px - mutual_information - beta * total_correlation - kld, axis=0
    )
