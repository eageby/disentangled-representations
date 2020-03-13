import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(
        self,
        f_phi,
        f_phi_mean,
        f_phi_log_var,
        f_theta,
        f_theta_mean,
        f_theta_log_var,
        objective,
        latents,
        **kwargs
    ):
        super(VAE, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()

        # Encoder
        self.f_phi = f_phi
        self.f_phi_mean = f_phi_mean
        self.f_phi_log_var = f_phi_log_var

        # Decoder
        self.f_theta = f_theta
        self.f_theta_mean = f_theta_mean
        self.f_theta_log_var = f_theta_log_var

        self.objective = objective

        self.latents = latents

    def build(self, input_shape):
        intermediate_shape = self.f_phi.compute_output_shape(input_shape)[1:]
        self.f_theta_dense = tf.keras.layers.Dense(
            tf.reduce_prod(intermediate_shape),
            activation='relu',
        )
        self.reshape_theta = tf.keras.layers.Reshape(
            intermediate_shape, name="ReshapeTheta"
        )

        self.reshape_output = tf.keras.layers.Reshape(
            input_shape[1:], name="ReshapeOutput"
        )

        super().build(input_shape)

    def sample(self, mean, log_var, training):
        if not training:
            return mean

        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)

        return mean + tf.exp(0.5 * log_var) * noise

    @tf.function
    def encode(self, x, training=False):
        x = self.f_phi(x)
        x = self.flatten(x)
        z_mean = self.f_phi_mean(x)
        z_log_var = self.f_phi_log_var(x)

        return z_mean, z_log_var

    @tf.function
    def decode(self, x):
        x = self.f_theta_dense(x)
        x = self.reshape_theta(x)
        x = self.f_theta(x)
        x_mean = self.f_theta_mean(x)
        x_log_var = self.f_theta_log_var(x)

        return self.reshape_output(x_mean), x_log_var

    def call(self, target, training=False):
        z_mean, z_log_var = self.encode(target)
        z = self.sample(z_mean, z_log_var, training)
        x_mean, x_log_var = self.decode(z)

        self.add_loss(self.objective(target, x_mean, x_log_var, z_mean, z_log_var))

        return x_mean, z, target
