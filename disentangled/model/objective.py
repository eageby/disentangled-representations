import tensorflow as tf

class BetaVAE():
    def __init__(self, gaussian):
        self.gaussian = gaussian

        if not gaussian:
            self.bce = tf.keras.losses.BinaryCrossentropy()

    def __call__(self, *args):
        return self.objective(*args)

    def kld(self, z_mean, z_log_var):
        inner = tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                ,axis=-1)
        return tf.reduce_mean(-0.5 * inner, axis=0)

    def log_likelihood(self, target, x_mean, x_log_var):
        return tf.reduce_mean(-0.5 * 
                   tf.reduce_sum(  
                       x_log_var + tf.square(target - x_mean) / tf.exp(x_log_var)
                   , axis=-1),
               axis=0)
        
    def objective(self, target, x_mean, x_log_var, z_mean, z_log_var):
        if self.gaussian:
            return self.log_likelihood(target, x_mean, x_log_var) + self.kld(z_mean, z_log_var)
        else:
            return self.bce(target, x_mean) + self.kld(z_mean, z_log_var)
