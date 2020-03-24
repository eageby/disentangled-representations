import math

import tensorflow as tf

_TOLERANCE = 1e-10

class _Objective(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(_Objective, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, *args):
        return self.objective(*[self.flatten(i) for i in args])


@tf.function
def log_likelihood_gaussian(target, x_mean, x_log_var):
    x_log_var = tf.clip_by_value(x_log_var, -_TOLERANCE, tf.float32.max)

    log_likelihood = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            x_log_var
            + tf.square(target - x_mean) / (tf.exp(x_log_var) + _TOLERANCE)
            + tf.math.log(2 * math.pi),
            axis=1,
        ),
        axis=0,
    )

    normalized_log_likelihood = log_likelihood + tf.reduce_mean(
        -0.5 * tf.reduce_sum(x_log_var + tf.math.log(2 * math.pi), axis=1), axis=0,
    )

    return log_likelihood, normalized_log_likelihood

def loglikelihood_bernoulli(target, x_mean):
    return tf.reduce_mean(
        tf.reduce_sum(
            target * tf.math.log(x_mean+_TOLERANCE) + (1 - target) * tf.math.log(1 - x_mean+_TOLERANCE),
            axis=1,
        ),
        axis=0,
    )


@tf.function
def kld_gaussian(z_mean, z_log_var):
    return tf.reduce_mean(
        -0.5
        * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1),
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
            log_likelihood, normalized_log_likelihood = log_likelihood_gaussian(
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

        self.add_metric(-log_likelihood, aggregation="mean", name="-loglikelihood")
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
            tf.math.log(discriminator_probability / ((1 - discriminator_probability) + _TOLERANCE))
        )

        try:
            tf.debugging.check_numerics(log_likelihood,'')
            tf.debugging.check_numerics(kld, '')
            tf.debugging.check_numerics(kld_discriminator, '')
        except:
            import pdb;pdb.set_trace()

        self.add_metric(-log_likelihood, aggregation="mean", name="-loglikelihood")
        self.add_metric(kld, aggregation="mean", name="kld1")
        self.add_metric(kld_discriminator, aggregation="mean", name="kld2")

        return -log_likelihood + kld + self.gamma * kld_discriminator
