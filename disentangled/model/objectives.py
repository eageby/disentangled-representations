import tensorflow as tf
import math

class BetaVAE:
    def __init__(self, gaussian=True, **kwargs):
        self.gaussian = gaussian

        if not gaussian:
            self.bce = tf.keras.losses.BinaryCrossentropy()

    def kld(self, z_mean, z_log_var):
        # return -0.5*tf.reduce_mean( (
        #     tf.reduce_sum(
        #         1 + z_log_var - tf.square(z_mean) , axis=1)
        #     - tf.reduce_sum(tf.math.exp(z_log_var))
        # ), axis=0)

        return tf.reduce_mean(
            -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            ),
            axis=0,
        )
    # KLD = 0.5 * T.sum(1 + log_sigma - mu**2 - T.exp(log_sigma), axis=1)

    def log_likelihood(self, target, x_mean, x_log_var):
        if not self.gaussian:
            return -self.bce(target, x_mean)

        return -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                 x_log_var + tf.square(target - x_mean) / tf.exp(x_log_var)
                 , axis=1)
            , axis=0)

            # logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
            #           0.5 * ((x - reconstructed_x)**2 / T.exp(log_sigma_decoder))).sum(axis=2).mean(axis=0)

    def objective(self, target, x_mean, x_log_var, z_mean, z_log_var):
        if self.gaussian:
            return -self.log_likelihood(target, x_mean, x_log_var) 
            # + self.kld(
            #      z_mean, z_log_var
            # )
        else:
            return self.bce(target, x_mean) + self.kld(z_mean, z_log_var)
            # return  self.kld(z_mean, z_log_var)
