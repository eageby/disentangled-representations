import tensorflow as tf
from .betavae import BetaVAE

import disentangled.model.networks as networks
import disentangled.utils
import disentangled.model.objectives as objectives

class SparseVAE(BetaVAE):
    def __init__(
        self,
        f_phi,
        f_theta,
        f_theta_mean,
        f_theta_log_var,
        latents,
        beta,
        gamma
    ):
        super().__init__(
            # Encoder
            f_phi=f_phi,
            f_phi_mean= tf.keras.layers.Dense( latents, activation=None, kernel_regularizer=tf.keras.regularizers.l1(1)),
            f_phi_log_var = tf.keras.layers.Dense( latents, activation=None, kernel_regularizer=tf.keras.regularizers.l1(1)),
            # Decoder
            f_theta=f_theta,
            f_theta_mean=f_theta_mean,
            f_theta_log_var=f_theta_log_var,
            objective=objectives.SparseVAE(beta, gamma),
            latents=latents,
            name='SparseVAE'
        )

    def train(self, data, learning_rate, iterations=100, **kwargs):
        data = data.take(int(iterations))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        progress = disentangled.utils.TrainingProgress(data, total=int(iterations))

        @tf.function
        def step(batch):
            with tf.GradientTape() as tape:
                z_mean, z_log_var = self.encode(batch)
                z = self.sample(z_mean, z_log_var, training=True)
               
                x_mean, x_log_var = self.decode(z)
                l1 = tf.reshape(tf.reduce_sum(self.losses), (1,-1))

                loss = self.objective(batch, x_mean, x_log_var, z_mean, z_log_var, l1)

            tf.debugging.check_numerics(loss, 'Loss is not valid')
            # Discriminator weights are assigned as not trainable in init
            grad = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grad, self.trainable_variables))
            metrics = {m.name: m.result() for m in self.metrics}
            return loss, metrics

        for batch in progress:        
            progress.update(*step(batch), interval=1)


class sparsevae_shapes3d(SparseVAE):
    def __init__(self, latents, beta, gamma, **kwargs):
        super().__init__(
            # Encoder
            f_phi=networks.conv_4,
            # Decoder
            f_theta=networks.conv_4_transpose,
            f_theta_mean=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation="sigmoid"
            ),
            f_theta_log_var=tf.keras.layers.Conv2DTranspose(
                3, kernel_size=(3, 3), strides=(1, 1), activation=None
            ),
            latents=latents,
            beta=beta,
            gamma=gamma
        )
