import disentangled.model.distributions as dist
import disentangled.model.networks as networks
import disentangled.model.objectives as objectives
import disentangled.utils
import gin.tf
import numpy as np
import tensorflow as tf

from .vae import VAE


@gin.configurable(module="model")
class FactorVAE(VAE):
    def __init__(self, discriminator, **kwargs):
        super().__init__(name="FactorVAE", **kwargs)

        self.discriminator_net = discriminator
        self.discriminator_net.trainable = False

    def discriminator(self, z):
        probabilities = self.discriminator_net(z)

        return tf.split(probabilities, 2, axis=-1)[0]

    @staticmethod
    def permute_dims(representation):
        representation = np.array(representation)
        
        for j in range(representation.shape[1]):
            permutation_index = tf.random.shuffle(tf.range(representation.shape[0]))        
            representation[:, j] = representation[permutation_index, j]

        return representation

    @gin.configurable(module="model.FactorVAE", blacklist=["data"])
    def train(
        self,
        data,
        optimizer,
        optimizer_discriminator,
        iterations,
        discriminator_loss,
        callbacks,
    ):
        data = data.batch(2)

        progress = disentangled.utils.TrainingProgress(
            data.take(int(iterations)), total=int(iterations)
        )

        [cb.set_model(self) for cb in callbacks]
        [cb.on_train_begin(progress.n) for cb in callbacks]

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

            logs = {m.name: m.result() for m in self.metrics}
            logs["loss"] = loss_psi

            return logs

        for batches in progress:
            [cb.on_train_batch_begin(progress.n) for cb in callbacks]
            logs = step(*batches)
            [cb.on_train_batch_end(progress.n, logs) for cb in callbacks]

            progress.update(logs.copy(), interval=1)

        [cb.on_train_end(progress.n) for cb in callbacks]
