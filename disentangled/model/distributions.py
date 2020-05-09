import math

import tensorflow as tf

_TOLERANCE = 1e-7


class Gaussian:
    def __init__(self, mean=0.0, log_var=0.0):
        self.mean = mean
        self.log_var = log_var

    @tf.function
    def log_likelihood(self, sample=0.0, mean=None, log_var=None):
        if mean is None:
            mean = self.mean

        if log_var is None:
            log_var = self.log_var

        return -0.5 * (
            log_var
            + tf.square(sample - mean) / (tf.exp(log_var) + _TOLERANCE)
            + tf.math.log(2 * math.pi)
        )

    @tf.function
    def kld(self, x_mean, x_log_var, y_mean=None, y_log_var=None):
        """Multivariate, diagonal Gaussians"""
        if y_mean is None:
            y_mean = self.mean

        if y_log_var is None:
            y_log_var = self.log_var

        x_log_var = tf.clip_by_value(
            x_log_var, tf.float32.min, tf.math.exp(
                tf.math.log(tf.float32.max)) - 1
        )

        return tf.reduce_sum(
            -0.5
            * (
                1 +
                x_log_var
                - y_log_var
                - tf.square(x_mean - y_mean) / (tf.math.exp(y_log_var))
                - tf.math.exp(x_log_var - y_log_var)
            ),
            axis=1,
        )

class Bernoulli:
    @tf.function
    def log_likelihood(self, target, mean):
        return target * tf.math.log(mean + _TOLERANCE)  \
            + (1 - target) * tf.math.log(1 - mean + _TOLERANCE)


class Laplacian:
    def init(self, mean=0.0, log_var=0.0):
        self.mean = mean
        self.log_var = log_var

    @tf.function
    def kld(self, x_mean, x_log_var, y_mean=None, y_log_var=None):
        """ Analytical Lower Bound"""

        return tf.reduce_mean(
            tf.reduce_sum(tf.math.exp(x_log_var) - x_log_var, axis=1), axis=0
        )
        # if y_mean is None:
        #     y_mean = self.mean
        # if y_log_var is None:
        #     y_log_var = self.log_var

        # x_log_var = tf.clip_by_value(
        #     x_log_var, tf.float32.min, tf.math.exp(tf.math.log(tf.float32.max)) - 1
        # )
        # y_log_var = tf.clip_by_value(
        #     y_log_var, tf.float32.min, tf.math.exp(tf.math.log(tf.float32.max)) - 1
        # )
        # return tf.reduce_sum(
        #         tf.math.exp(x_log_var) / (tf.math.exp(y_log_var) - _TOLERANCE) -
        #         + tf.math.abs(x_mean - y_mean) / (tf.math.exp(y_log_var) + _TOLERANCE)
        #         x_log_var - y_log_var - 1
        #     , axis=-1)

