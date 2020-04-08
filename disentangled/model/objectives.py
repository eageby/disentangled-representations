import math

import tensorflow as tf

_TOLERANCE = 1e-7


class _Objective(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(_Objective, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, *args):
        return self.objective(*[self.flatten(i) for i in args])


@tf.function
def logsumexp(data, axis=-1, keepdims=False):
    m = tf.math.reduce_max(data, axis=axis, keepdims=True)

    value = tf.math.log(tf.math.reduce_sum(tf.math.exp(data - m))) + m
    if not keepdims:
        return tf.squeeze(value)
    return value
        
@tf.function
def log_likelihood_gaussian(x_mean, x_log_var):
    return -0.5 * (
        x_log_var
        + tf.square(x_mean) / (tf.exp(x_log_var) + _TOLERANCE)
        + tf.math.log(2 * math.pi)
    )

@tf.function
def log_likelihood_gaussian_objective(target, x_mean, x_log_var):
    x_log_var = tf.clip_by_value(
        x_log_var, tf.float32.min, tf.math.exp(tf.math.log(tf.float32.max)) - 1
    )

    log_likelihood = tf.reduce_mean(
        tf.reduce_sum(log_likelihood_gaussian(
            target - x_mean, x_log_var), axis=1),
        axis=0,
    )

    normalized_log_likelihood = log_likelihood + tf.reduce_mean(
        -0.5 * tf.reduce_sum(x_log_var + tf.math.log(2 * math.pi), axis=1), axis=0
    )

    return log_likelihood, normalized_log_likelihood


def loglikelihood_bernoulli(target, x_mean):
    return tf.reduce_mean(
        tf.reduce_sum(
            target * tf.math.log(x_mean + _TOLERANCE)
            + (1 - target) * tf.math.log(1 - x_mean + _TOLERANCE),
            axis=1,
        ),
        axis=0,
    )


@tf.function
def kld_gaussian(z_mean, z_log_var):
    z_log_var = tf.clip_by_value(
        z_log_var, tf.float32.min, tf.math.exp(tf.math.log(tf.float32.max)) - 1
    )

    return tf.reduce_mean(
        -0.5
        * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) -
                        tf.exp(z_log_var), axis=1),
        axis=0,
    )


class BetaVAE(_Objective):
    def __init__(self, beta=1, gaussian=False, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.beta = beta
        self.gaussian = gaussian

    @tf.function
    def objective(self, target, x_mean, x_log_var, z_mean, z_log_var):
        if self.gaussian:
            log_likelihood, normalized_log_likelihood = log_likelihood_gaussian_objective(
                target, x_mean, x_log_var
            )

            self.add_metric(
                normalized_log_likelihood,
                aggregation="mean",
                name="Normalized loglikelihood",
            )
        else:
            log_likelihood = loglikelihood_bernoulli(target, x_mean)

        kld = kld_gaussian(z_mean, z_log_var)

        self.add_metric(-log_likelihood, aggregation="mean",
                        name="-loglikelihood")
        self.add_metric(kld, aggregation="mean", name="kld")

        return -log_likelihood + self.beta * kld


class FactorVAE(_Objective):
    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def objective(
        self, target, x_mean, x_log_var, z_mean, z_log_var, discriminator_probability
    ):

        log_likelihood = loglikelihood_bernoulli(target, x_mean)
        kld = kld_gaussian(z_mean, z_log_var)

        kld_discriminator = tf.reduce_mean(
            tf.math.log(
                discriminator_probability
                / ((1 - discriminator_probability) + _TOLERANCE)
            )
        )

        self.add_metric(-log_likelihood, aggregation="mean",
                        name="-loglikelihood")
        self.add_metric(kld, aggregation="mean", name="kld1")
        self.add_metric(kld_discriminator, aggregation="mean", name="kld2")

        return -log_likelihood + kld + self.gamma * kld_discriminator

    @staticmethod
    def discriminator(p_z, p_permuted):
        return -tf.reduce_mean(
            tf.math.log(p_z + _TOLERANCE) +
            tf.math.log(p_permuted + _TOLERANCE)
        )


class Beta_TCVAE(_Objective):
    def __init__(self, beta, dataset_size=1000, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.dataset_size = dataset_size

    def objective(self, target, x_mean, x_log_var, z_mean, z_log_var):
        log_nm = tf.math.log(float(target.shape[0] * self.dataset_size))

        log_px = loglikelihood_bernoulli(target, x_mean)

        log_pz = tf.reduce_sum(log_likelihood_gaussian(z_mean, z_log_var), axis=1)
        log_qzx = tf.reduce_sum(
            log_likelihood_gaussian(z_mean, z_log_var), axis=1)

        log_qz_matrix = log_likelihood_gaussian(
            tf.expand_dims(z_mean, axis=1), tf.expand_dims(z_log_var, axis=0)
        )

        log_qz_prod = tf.reduce_sum(
            logsumexp(log_qz_matrix, axis=1) - log_nm
            , axis=1
        )

        log_qz = logsumexp(tf.reduce_sum(log_qz_matrix, axis=2), axis=1) - log_nm

        mutual_information = log_qzx - log_qz
        total_correlation = log_qz - log_qz_prod
        kld = log_qz_prod - log_pz

        self.add_metric(-log_px, aggregation="mean",
                        name="-loglikelihood")
        self.add_metric(mutual_information, aggregation="mean", name="info")
        self.add_metric(kld, aggregation="mean", name="kld")
        self.add_metric(total_correlation, aggregation="mean", name="tc")

        return -tf.reduce_mean(log_px - mutual_information - self.beta * total_correlation - kld, axis=0)
