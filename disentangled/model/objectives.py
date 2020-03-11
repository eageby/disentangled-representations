import math

import tensorflow as tf


class _Objective(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(_Objective, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input_):
        return self.objective(*[self.flatten(i) for i in input_])

@tf.function
def log_likelihood_gaussian(target, x_mean, x_log_var):
    log_likelihood = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            x_log_var
            + tf.square(target - x_mean) / tf.exp(x_log_var)
            + tf.math.log(2 * math.pi),
            axis=1,
        ),
        axis=0,
    )

    normalized_log_likelihood = log_likelihood + tf.reduce_mean(
        -0.5 * tf.reduce_sum(x_log_var + tf.math.log(2 * math.pi), axis=1),
        axis=0,
    )

    return log_likelihood, normalized_log_likelihood

@tf.function
def loglikelihood_bernoulli(target, x_mean, x_log_var):
    return tf.reduce_mean(
        tf.reduce_sum(
            target * tf.math.log(x_mean) + (1 - target) * tf.math.log(1 - x_mean),
            axis=1,
        ),
        axis=0,
    )

@tf.function
def kld_gaussian(z_mean, z_log_var):
    return tf.reduce_mean(
        -0.5
        * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        ),
        axis=0,
    )

class BetaVAE(_Objective):
    def __init__(self, beta=1, gaussian=False, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.beta = beta
        self.gaussian = gaussian
        import pdb;pdb.set_trace()

    @tf.function
    def objective(self, target, x_mean, x_log_var, z_mean, z_log_var):
        if self.gaussian: 
            log_likelihood, normalized_log_likelihood = log_likelihood_gaussian(target, x_mean, x_log_var)

            self.add_metric(
                normalized_log_likelihood,
                aggregation="mean",
                name="Normalized loglikelihood",
            )
        else:
            log_likelihood = loglikelihood_bernoulli(target, x_mean, x_log_var)

        kld = kld_gaussian(z_mean, z_log_var)

        self.add_metric(-log_likelihood, aggregation="mean", name="-loglikelihood")
        self.add_metric(kld, aggregation="mean", name="kld")

        return -log_likelihood + self.beta*kld


class FactorVAE(_Objective):
    def __init__(self, gamma, **kwargs):
        super().__init__(self, **kwargs) 
        self.gamma = gamma

