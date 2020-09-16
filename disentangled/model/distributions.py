import math
import gin
import tensorflow as tf

@gin.configurable
class Gaussian:
    def __init__(self, mean=0.0, log_var=0.0, tolerance=0.0):
        self.mean = mean
        self.log_var = log_var
        self.tolerance = tolerance

    @tf.function
    def log_likelihood(self, sample=0.0, mean=None, log_var=None):
        if mean is None:
            mean = self.mean

        if log_var is None:
            log_var = self.log_var
        return -0.5 * (
            log_var
            + tf.square(sample - mean) / (tf.exp(log_var) + self.tolerance)
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
            x_log_var, tf.float32.min, tf.math.exp(tf.math.log(tf.float32.max)) - 1
        )

        return -0.5 * (
            1
            + x_log_var
            - y_log_var
            - tf.square(x_mean - y_mean) / (tf.math.exp(y_log_var))
            - tf.math.exp(x_log_var - y_log_var)
        )


@gin.configurable
class Bernoulli:
    def __init__(self, tolerance=0.0):
        self.tolerance = tolerance

    @tf.function
    def log_likelihood(self, target, mean, *args):
        return target * tf.math.log(mean + self.tolerance) + (1 - target) * tf.math.log(
            1 - mean + self.tolerance
        )


@gin.configurable
class Laplacian:
    def __init__(self, location=0.0, log_scale=0.0, tolerance=0.0):
        self.location = location
        self.log_scale = log_scale
        self.tolerance = tolerance

    # @tf.function
    # def kld(self, x_location, x_log_scale, y_location=None, y_log_scale=None):
    #     """ Analytical Lower Bound"""

    #     return tf.math.exp(x_log_scale) - x_log_scale - 1

    @tf.function
    def log_likelihood(self, sample=0.0, location=None, log_scale=None):
        if location is None:
            location = self.location

        if log_scale is None:
            log_scale = self.log_scale

        return -log_scale - tf.math.log(2.0) - tf.math.abs(sample - location) / tf.math.exp(log_scale)

    @tf.function
    def kld(self, x_location, x_log_scale, y_location=None, y_log_scale=None):
        """ Analytical Lower Bound"""

        if y_location is None:
            y_location = self.location

        if y_log_scale is None:
            y_log_scale = self.log_scale

        x_log_scale = tf.clip_by_value(
            x_log_scale, tf.float32.min, tf.math.exp(tf.math.log(tf.float32.max)) - 1
        )
        y_log_scale = tf.clip_by_value(
            y_log_scale, tf.float32.min, tf.math.exp(tf.math.log(tf.float32.max)) - 1
        )

        return (
            tf.math.exp(0.5 * x_log_scale)
            / (tf.math.exp(0.5 * y_log_scale) - self.tolerance)
            - 0.5 * x_log_scale
            + 0.5 * y_log_scale
            - 1
            + tf.math.abs(x_location - y_location)
            / (tf.math.exp(0.5 * y_log_scale) + self.tolerance)
        )
