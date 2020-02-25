import tensorflow as tf

class _Objective(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(_Objective, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input_):
        return self.objective(*[self.flatten(i) for i in input_])

class BetaVAE(_Objective):
    def __init__(self, gaussian, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        self.gaussian = gaussian

        if not gaussian:
            self.bce = tf.keras.losses.BinaryCrossentropy()

    def kld(self, z_mean, z_log_var):
        return tf.reduce_mean((1 / 2) * (
            tf.reduce_sum(
                tf.square(z_mean) - z_log_var - 1, axis=1)
            + tf.reduce_sum(tf.math.exp(z_log_var))
        ), axis=0)
        # return -0.5 * tf.reduce_mean(
        #     tf.reduce_sum(
        #         1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        #     ),
        #     axis=0,
        # )

    def log_likelihood(self, target, x_mean, x_log_var):
        return -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                x_log_var + tf.square(target - x_mean) / tf.exp(x_log_var), axis=-1
            ),
            axis=0,
        )

    def objective(self, target, x_mean, x_log_var, z_mean, z_log_var):
        if self.gaussian:
            return -self.log_likelihood(target, x_mean, x_log_var) 
            # + self.kld(
            #      z_mean, z_log_var
            # )
        else:
            return self.bce(target, x_mean) + self.kld(z_mean, z_log_var)
            # return  self.kld(z_mean, z_log_var)

