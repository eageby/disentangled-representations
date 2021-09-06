import disentangled.utils
import gin.tf
import tensorflow as tf


@gin.configurable
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
            tf.reduce_prod(intermediate_shape), activation="relu"
        )
        self.reshape_theta = tf.keras.layers.Reshape(
            intermediate_shape, name="ReshapeTheta"
        )

        self.reshape_output = tf.keras.layers.Reshape(
            input_shape[1:], name="ReshapeOutput"
        )

        super().build(input_shape)

    @tf.function
    def sample(self, mean, log_var, training=False):
        if not training:
            return mean

        noise = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0)
        return mean + tf.exp(0.5 * log_var) * noise

    @tf.function
    def encode(self, x):
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

    @tf.function
    def call(self, target, training=False):
        z_mean, z_log_var = self.encode(target)
        z = self.sample(z_mean, z_log_var, training)
        x_mean, x_log_var = self.decode(z)

        return x_mean, z, target

    @gin.configurable(module="VAE", denylist=["iterations", "callbacks"])
    def train(self, data, optimizer, iterations, callbacks):
        data = data.take(int(iterations))
        progress = disentangled.utils.TrainingProgress(data, total=int(iterations))

        [cb.set_model(self) for cb in callbacks]

        [cb.on_train_begin(progress.n) for cb in callbacks]

        @tf.function
        def step(batch):
            with tf.GradientTape() as tape:
                z_mean, z_log_var = self.encode(batch)
                z = self.sample(z_mean, z_log_var, training=True)

                x_mean, x_log_var = self.decode(z)

                loss = self.objective(batch, x_mean, x_log_var, z, z_mean, z_log_var)

            tf.debugging.check_numerics(loss, "Loss is not valid")

            grad = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.trainable_variables))
            logs = {m.name: m.result() for m in self.metrics}
            logs["loss"] = loss

            return logs

        for batch in progress:
            [cb.on_train_batch_begin(progress.n) for cb in callbacks]
            logs = step(batch)
            [cb.on_train_batch_end(progress.n, logs) for cb in callbacks]
            progress.update(logs, interval=1)

        [cb.on_train_end(progress.n) for cb in callbacks]
