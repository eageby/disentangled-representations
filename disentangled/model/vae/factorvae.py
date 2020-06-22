import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.objectives as objectives
import disentangled.utils
import numpy as np
import tensorflow as tf
import gin.tf

from .vae import VAE


@gin.configurable(module='model')
class FactorVAE(VAE):
    def __init__(self, discriminator, **kwargs):
        super().__init__(
            name="FactorVAE",
            **kwargs
        )

        self.discriminator_net = discriminator
        self.discriminator_net.trainable = False

    def discriminator(self, z):
        probabilities = self.discriminator_net(z)

        return tf.split(probabilities, 2, axis=-1)[0]

    @staticmethod
    def permute_dims(representation):
        representation = np.array(representation)

        for j in range(representation.shape[1]):
            permutation_index = np.random.permutation(representation.shape[0])
            representation[:, j] = representation[permutation_index, j]

        return representation

    @gin.configurable(module='model.FactorVAE', blacklist=['data'])
    def train(
        self, data, optimizer=gin.REQUIRED, optimizer_discriminator=gin.REQUIRED, iterations=gin.REQUIRED, discriminator_loss=gin.REQUIRED
    ):
        data = data.batch(2)

        progress = disentangled.utils.TrainingProgress(
            data.take(int(iterations)), total=int(iterations)
        )

        @tf.function
        def step(batch_theta, batch_psi):
            with tf.GradientTape() as tape:
                z_mean, z_log_var = self.encode(batch_theta)
                z = self.sample(z_mean, z_log_var, training=True)

                p_z = self.discriminator(z)

                x_mean, x_log_var = self.decode(z)

                loss_theta = self.objective(
                    batch_theta, x_mean, x_log_var, z_mean, z_log_var, p_z
                )
                tf.debugging.check_numerics(loss_theta, "loss is invalid")

            # Discriminator weights are assigned as not trainable in init
            grad_theta = tape.gradient(loss_theta, self.trainable_variables)
            optimizer.apply_gradients(
                zip(grad_theta, self.trainable_variables))

            # Updating Discriminator
            with tf.GradientTape() as tape:
                z_mean, z_log_var = self.encode(batch_psi)
                z = self.sample(z_mean, z_log_var, training=True)

                z_permuted = tf.py_function(
                    self.permute_dims, inp=[z], Tout=tf.float32)
                z_permuted.set_shape(z.shape)

                p_permuted = self.discriminator(z_permuted)

                loss_psi = discriminator_loss(p_z, p_permuted)

            grad_psi = tape.gradient(
                loss_psi, self.discriminator_net.variables)
            optimizer_discriminator.apply_gradients(
                zip(grad_psi, self.discriminator_net.variables)
            )

            metrics = {m.name: m.result() for m in self.metrics}

            return loss_theta, metrics

        for batches in progress:
            progress.update(*step(*batches), interval=1)


class factorvae_shapes3d(FactorVAE):
    def __init__(self, latents, gamma, **kwargs):
        super().__init__(
            f_phi=networks.conv_4,
            f_phi_mean=tf.keras.layers.Dense(latents, activation=None),
            f_phi_log_var=tf.keras.layers.Dense(latents, activation=None),
            f_theta=networks.conv_4_transpose,
            f_theta_mean=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
            ),
            f_theta_log_var=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation=None
            ),
            latents=latents,
            gamma=gamma,
            discriminator=tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        1000, activation=tf.nn.leaky_relu, input_shape=(latents,)
                    ),
                    tf.keras.layers.Dense(1000, activation=tf.nn.leaky_relu),
                    tf.keras.layers.Dense(1000, activation=tf.nn.leaky_relu),
                    tf.keras.layers.Dense(1000, activation=tf.nn.leaky_relu),
                    tf.keras.layers.Dense(1000, activation=tf.nn.leaky_relu),
                    tf.keras.layers.Dense(1000, activation=tf.nn.leaky_relu),
                    tf.keras.layers.Dense(2, activation="softmax"),
                ]
            ),
        )
